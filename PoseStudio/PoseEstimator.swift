import Foundation
import AVFoundation
import Vision
import ImageIO
import Combine
import CoreGraphics

// MARK: - 数据模型

/// 单个关节
struct Joint {
    let index: Int
    let name: String
    let position: CGPoint   // Vision 归一化坐标，原点在左下
    let confidence: Float
}

/// 一个人
struct Pose: Identifiable {
    let id: Int
    var points: [Joint?]        // 固定 19 项，索引对应 Self.jointNames 顺序
    var confidence: Float
    var boundingBox: CGRect     // Vision 归一化坐标（原点左下）
}

/// 一颗物理镜头。切换它就是真正的光学变焦（换镜头），不是裁切放大。
struct LensOption: Identifiable, Hashable {
    let id: String
    let label: String
    let deviceType: AVCaptureDevice.DeviceType
}

// MARK: - 平滑滤波

/// One Euro Filter —— 姿态关键点平滑的标准做法。
///
/// 比"滑动平均"好在于它是**自适应**的：
///   * 手不动时 → 截止频率低 → 强滤波，把抖动压掉
///   * 快速挥手时 → 截止频率自动升高 → 弱滤波，不产生拖影延迟
/// 原理：先用低通估出速度，再按速度动态决定平滑强度。
final class OneEuroFilter {
    private var xPrev: Double?
    private var dxPrev: Double = 0
    private var tPrev: Double?

    private let minCutoff: Double
    private let beta: Double
    private let dCutoff: Double

    init(minCutoff: Double = 1.0, beta: Double = 0.02, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    func reset() {
        xPrev = nil
        dxPrev = 0
        tPrev = nil
    }

    private func alpha(_ cutoff: Double, _ dt: Double) -> Double {
        let tau = 1.0 / (2.0 * Double.pi * max(cutoff, 0.0001))
        return 1.0 / (1.0 + tau / max(dt, 0.0001))
    }

    func filter(_ x: Double, at time: Double) -> Double {
        guard let xp = xPrev, let tp = tPrev else {
            xPrev = x
            tPrev = time
            return x
        }
        var dt = time - tp
        if dt <= 0 || dt > 0.5 { dt = 1.0 / 30.0 }
        tPrev = time

        // 先估速度并低通
        let dx = (x - xp) / dt
        let aD = alpha(dCutoff, dt)
        let dxHat = aD * dx + (1.0 - aD) * dxPrev
        dxPrev = dxHat

        // 速度越大，截止频率越高（越不平滑，越跟手）
        let cutoff = minCutoff + beta * abs(dxHat)
        let a = alpha(cutoff, dt)
        let xHat = a * x + (1.0 - a) * xp
        xPrev = xHat
        return xHat
    }
}

/// 一个关节的 x/y 两个滤波器
final class JointFilter {
    let fx: OneEuroFilter
    let fy: OneEuroFilter
    init(minCutoff: Double, beta: Double) {
        fx = OneEuroFilter(minCutoff: minCutoff, beta: beta)
        fy = OneEuroFilter(minCutoff: minCutoff, beta: beta)
    }
    func reset() { fx.reset(); fy.reset() }
}

// MARK: - 姿态推理

final class PoseEstimator: NSObject, ObservableObject {

    // 19 个关节
    static let jointNames: [String] = [
        "鼻", "左眼", "右眼", "左耳", "右耳",
        "颈", "左肩", "右肩", "左肘", "右肘",
        "左腕", "右腕", "左髋", "右髋", "左膝",
        "右膝", "左踝", "右踝", "根",
    ]

    // 骨架连线：(起点索引, 终点索引, 分组)
    static let connections: [(Int, Int, String)] = [
        (0, 1, "head"), (0, 2, "head"), (1, 3, "head"), (2, 4, "head"), (5, 0, "head"),
        (5, 6, "torso"), (5, 7, "torso"), (6, 7, "torso"),
        (6, 12, "torso"), (7, 13, "torso"), (12, 13, "torso"),
        (6, 8, "armL"), (8, 10, "armL"),
        (7, 9, "armR"), (9, 11, "armR"),
        (12, 14, "legL"), (14, 16, "legL"),
        (13, 15, "legR"), (15, 17, "legR"),
        (5, 18, "torso"), (12, 18, "torso"), (13, 18, "torso"),
    ]

    static let faceJointIndices: Set<Int> = [1, 2, 3, 4]

    // MARK: 发布给界面的状态

    @Published var poses: [Pose] = []
    @Published var fps: Double = 0
    @Published var inferenceMS: Double = 0
    @Published var videoSize: CGSize = .zero
    @Published var isRunning = false
    @Published var permissionDenied = false
    @Published var statusText: String = "未启动"
    @Published var orientationLabel: String = "探测中…"

    /// 当前可用的物理镜头（会随前后置切换而变化）
    @Published var availableLenses: [LensOption] = []
    @Published var currentLensID: String = ""
    @Published var useFrontCamera: Bool = false
    /// 叠加层是否需要水平镜像（前置摄像头时预览是镜像的）
    @Published var overlayMirrored: Bool = false

    // 由界面同步进来的参数（避免跨线程读 ObservableObject）
    var detectionConfidence: Float = 0.30
    var maxPeople: Int = 1
    var fpsLimit: Double = 30
    var use3DModel: Bool = false

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "posestudio.session")
    private let videoQueue = DispatchQueue(label: "posestudio.video")
    private let request2D = VNDetectHumanBodyPoseRequest()
    private let request3D = VNDetectHumanBodyPose3DRequest()
    private var videoOutput: AVCaptureVideoDataOutput?
    private var currentInput: AVCaptureDeviceInput?
    private var currentDevice: AVCaptureDevice?
    private var front = false
    private var lensID = "wide"
    private var configured = false

    private var frameCount = 0
    private var lastFPSAt = CFAbsoluteTimeGetCurrent()
    private var lastProcessAt: CFAbsoluteTime = 0

    // ---- 平滑（One Euro）----
    var smoothingEnabled: Bool = true
    /// 0 = 轻，1 = 重
    var smoothingStrength: Double = 0.5
    private var jointFilters: [[JointFilter]] = []

    // ---- 方向自动探测 ----
    private let orientationList: [CGImagePropertyOrientation] = [.up, .right, .down, .left]
    private var probeScore = [0, 0, 0, 0]
    private var probeCursor = 0
    private var probeFrames = 0
    private var lockedOrientationIndex: Int? = nil

    private static let vision2DJoints: [(VNHumanBodyPoseObservation.JointName, Int)] = [
        (.nose, 0), (.leftEye, 1), (.rightEye, 2), (.leftEar, 3), (.rightEar, 4),
        (.neck, 5), (.leftShoulder, 6), (.rightShoulder, 7),
        (.leftElbow, 8), (.rightElbow, 9), (.leftWrist, 10), (.rightWrist, 11),
        (.leftHip, 12), (.rightHip, 13), (.leftKnee, 14), (.rightKnee, 15),
        (.leftAnkle, 16), (.rightAnkle, 17), (.root, 18),
    ]

    /// 3D 姿态模型**没有面部和颈部关节**（只有 13 个身体关节），
    /// 所以这里只映射存在的那些；头部的圆圈由叠加层用双肩推算。
    private static let vision3DJoints: [(VNHumanBodyPose3DObservation.JointName, Int)] = [
        (.leftShoulder, 6), (.rightShoulder, 7),
        (.leftElbow, 8), (.rightElbow, 9), (.leftWrist, 10), (.rightWrist, 11),
        (.leftHip, 12), (.rightHip, 13), (.leftKnee, 14), (.rightKnee, 15),
        (.leftAnkle, 16), (.rightAnkle, 17), (.root, 18),
    ]

    static func orientationName(_ o: CGImagePropertyOrientation) -> String {
        switch o {
        case .up: return "0°"
        case .right: return "90°"
        case .down: return "180°"
        default: return "270°"
        }
    }

    // MARK: 权限 + 启动

    func start(front isFront: Bool) {
        front = isFront
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            permissionDenied = false
            boot()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted { self.permissionDenied = false; self.boot() }
                    else { self.permissionDenied = true; self.statusText = "摄像头权限被拒绝" }
                }
            }
        default:
            permissionDenied = true
            statusText = "请在 设置 → 隐私 → 相机 中允许本应用"
        }
    }

    private func boot() {
        resetOrientationProbe()
        statusText = "启动中…"
        sessionQueue.async {
            self.configureIfNeeded()
            if !self.session.isRunning { self.session.startRunning() }
            DispatchQueue.main.async {
                self.isRunning = true
                self.useFrontCamera = self.front
                self.overlayMirrored = self.front
                self.statusText = (self.front ? "前置" : "后置") + " · " + self.currentLensLabel()
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
            DispatchQueue.main.async {
                self.isRunning = false
                self.statusText = "已停止"
                self.poses = []
            }
        }
    }

    // MARK: 镜头发现（光学变焦的关键）

    private func discoverLenses(position: AVCaptureDevice.Position) -> [LensOption] {
        let types: [AVCaptureDevice.DeviceType] = [
            .builtInUltraWideCamera, .builtInWideAngleCamera, .builtInTelephotoCamera,
        ]
        let discovery = AVCaptureDevice.DiscoverySession(deviceTypes: types,
                                                         mediaType: .video,
                                                         position: position)
        var out: [LensOption] = []
        for device in discovery.devices {
            let key: String
            let label: String
            switch device.deviceType {
            case .builtInUltraWideCamera: key = "ultrawide"; label = "0.5×"
            case .builtInTelephotoCamera: key = "tele";      label = "长焦"
            default:                      key = "wide";      label = "1×"
            }
            if !out.contains(where: { $0.id == key }) {
                out.append(LensOption(id: key, label: label, deviceType: device.deviceType))
            }
        }
        let order = ["ultrawide": 0, "wide": 1, "tele": 2]
        out.sort { (order[$0.id] ?? 9) < (order[$1.id] ?? 9) }
        return out
    }

    private func currentLensLabel() -> String {
        if let l = availableLenses.first(where: { $0.id == lensID }) { return l.label }
        return lensID
    }

    private func deviceForCurrentLens() -> AVCaptureDevice? {
        let position: AVCaptureDevice.Position = front ? .front : .back
        let type = availableLenses.first(where: { $0.id == lensID })?.deviceType
            ?? .builtInWideAngleCamera
        if let d = AVCaptureDevice.default(type, for: .video, position: position) { return d }
        return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
    }

    // MARK: 会话配置

    private func configureIfNeeded() {
        guard !configured else { return }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        let position: AVCaptureDevice.Position = front ? .front : .back
        let lenses = discoverLenses(position: position)
        if !lenses.isEmpty && !lenses.contains(where: { $0.id == lensID }) {
            lensID = lenses.first(where: { $0.id == "wide" })?.id ?? lenses[0].id
        }

        guard let device = deviceForCurrentLens(),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.statusText = "找不到可用摄像头" }
            return
        }
        session.addInput(input)
        currentInput = input
        currentDevice = device

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        output.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(output) {
            session.addOutput(output)
            videoOutput = output
        }

        applyConnectionSettings(dataMirrored: false)
        session.commitConfiguration()

        applyFrameRate(device, fps: fpsLimit)

        configured = true
        DispatchQueue.main.async {
            self.availableLenses = lenses
            self.currentLensID = self.lensID
        }
    }

    /// 方向 + 镜像。
    ///
    /// 镜像这点必须完全确定：数据输出**永不镜像**，预览层前置时镜像，
    /// 叠加层在前置时把 x 翻过来。三处显式对齐，就不存在"谁自动谁没自动"的错位。
    private func applyConnectionSettings(dataMirrored: Bool) {
        guard let conn = videoOutput?.connection(with: .video) else { return }
        if conn.isVideoRotationAngleSupported(90) {
            conn.videoRotationAngle = 90
        }
        if conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = dataMirrored
        }
    }

    /// 真正的帧率控制：改采集设备本身的帧率。
    /// 之前只做了软件丢帧（只能降不能升），所以往上调完全没反应。
    private func applyFrameRate(_ device: AVCaptureDevice, fps: Double) {
        // 注意：AVFrameRateRange 的 min/maxFrameRate 是 Double，不是 Float
        let target = min(max(fps, 1), 120)
        do {
            try device.lockForConfiguration()
            let ranges = device.activeFormat.videoSupportedFrameRateRanges
            // 只有落在支持区间内才去设，否则会抛 ObjC 异常（Swift 抓不住）
            var supported = false
            for r in ranges where target >= r.minFrameRate && target <= r.maxFrameRate {
                supported = true
                break
            }
            if supported {
                let dur = CMTime(value: 1, timescale: CMTimeScale(target.rounded()))
                device.activeVideoMinFrameDuration = dur
                device.activeVideoMaxFrameDuration = dur
            }
            device.unlockForConfiguration()
        } catch {
            // 设不了就算了，软件跳帧仍然生效
        }
    }

    func updateFrameRate(_ fps: Double) {
        sessionQueue.async {
            guard let d = self.currentDevice else { return }
            self.applyFrameRate(d, fps: fps)
        }
    }

    // MARK: 切换摄像头 / 镜头

    func switchCamera(front isFront: Bool) {
        front = isFront
        resetOrientationProbe()
        sessionQueue.async {
            self.rebuildInput()
        }
    }

    func switchLens(_ id: String) {
        lensID = id
        resetOrientationProbe()
        sessionQueue.async {
            self.rebuildInput()
        }
    }

    private func rebuildInput() {
        let position: AVCaptureDevice.Position = front ? .front : .back
        let lenses = discoverLenses(position: position)
        if !lenses.isEmpty && !lenses.contains(where: { $0.id == lensID }) {
            lensID = lenses.first(where: { $0.id == "wide" })?.id ?? lenses[0].id
        }

        session.beginConfiguration()
        if let old = currentInput {
            session.removeInput(old)
            currentInput = nil
        }
        guard let device = deviceForCurrentLens(),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        currentInput = input
        currentDevice = device
        applyConnectionSettings(dataMirrored: false)
        session.commitConfiguration()
        applyFrameRate(device, fps: fpsLimit)

        DispatchQueue.main.async {
            self.availableLenses = lenses
            self.currentLensID = self.lensID
            self.overlayMirrored = self.front
            self.statusText = (self.front ? "前置" : "后置") + " · " + self.currentLensLabel()
        }
    }

    /// 重新开始方向探测
    func resetOrientationProbe() {
        lockedOrientationIndex = nil
        probeScore = [0, 0, 0, 0]
        probeCursor = 0
        probeFrames = 0
        DispatchQueue.main.async { self.orientationLabel = "探测中…" }
    }

    // MARK: 每帧推理

    fileprivate func handle(pixelBuffer: CVPixelBuffer) {
        // 软件帧率兜底（设备帧率设不上去时仍然有效）
        let now0 = CFAbsoluteTimeGetCurrent()
        if fpsLimit >= 1 {
            let minInterval = 1.0 / min(max(fpsLimit, 1.0), 120.0)
            if now0 - lastProcessAt < minInterval { return }
            lastProcessAt = now0
        }

        let bufferW = CVPixelBufferGetWidth(pixelBuffer)
        let bufferH = CVPixelBufferGetHeight(pixelBuffer)

        // 方向：自动探测，不再猜
        let orientationIndex = lockedOrientationIndex ?? probeCursor
        let visionOrientation = orientationList[orientationIndex]
        let swapped = (visionOrientation == .left || visionOrientation == .right)
        let orientedW = swapped ? bufferH : bufferW
        let orientedH = swapped ? bufferW : bufferH

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: visionOrientation,
                                            options: [:])
        let t0 = CFAbsoluteTimeGetCurrent()

        var found: [Pose] = []
        if use3DModel {
            do { try handler.perform([request3D]) } catch { return }
            if let observations = request3D.results {
                for (i, obs) in observations.prefix(max(1, maxPeople)).enumerated() {
                    if let pose = Self.buildPose3D(obs, index: i) { found.append(pose) }
                }
            }
        } else {
            do { try handler.perform([request2D]) } catch { return }
            if let observations = request2D.results {
                for (i, obs) in observations.prefix(max(1, maxPeople)).enumerated() {
                    if let pose = Self.buildPose2D(obs, index: i,
                                                   minConfidence: detectionConfidence) {
                        found.append(pose)
                    }
                }
            }
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        // 探测阶段：累计每个方向的命中数，够了就锁定
        var lockedNow: Int? = nil
        if lockedOrientationIndex == nil {
            probeScore[orientationIndex] += found.count
            probeFrames += 1
            probeCursor = (probeCursor + 1) % orientationList.count
            if probeFrames >= orientationList.count * 2 {
                probeFrames = 0
                var bestIdx = 0
                var bestVal = 0
                for (i, v) in probeScore.enumerated() where v > bestVal {
                    bestVal = v
                    bestIdx = i
                }
                if bestVal > 0 {
                    lockedOrientationIndex = bestIdx
                    lockedNow = bestIdx
                }
            }
        }

        frameCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        var newFPS: Double = 0
        if now - lastFPSAt >= 0.5 {
            newFPS = Double(frameCount) / (now - lastFPSAt)
            frameCount = 0
            lastFPSAt = now
        }
        let lockToPublish = lockedNow
        let labelToPublish = Self.orientationName(visionOrientation)
        // 抖动就出在这一步之前：原始关键点逐帧跳，过一遍 One Euro 再发布
        let smoothedPoses = smooth(found, at: now)

        DispatchQueue.main.async {
            self.poses = smoothedPoses
            self.inferenceMS = elapsed
            if let _ = lockToPublish {
                self.orientationLabel = labelToPublish + " 已锁定"
            } else if self.lockedOrientationIndex == nil {
                self.orientationLabel = "探测中（" + labelToPublish + "）"
            }
            let newSize = CGSize(width: orientedW, height: orientedH)
            if self.videoSize != newSize { self.videoSize = newSize }
            if newFPS > 0 { self.fps = newFPS }
        }
    }

    // MARK: 平滑

    /// 对每个关节的归一化坐标做 One Euro 滤波，再用平滑后的点重算外接框。
    /// 滤波器是按「人索引 + 关节槽位」长期持有的，跨帧才有意义。
    private func smooth(_ input: [Pose], at time: Double) -> [Pose] {
        guard smoothingEnabled else {
            if !jointFilters.isEmpty { jointFilters.removeAll() }
            return input
        }

        // 强度 -> 参数：强度越大，静止时截止频率越低（滤波越重）
        let s = min(max(smoothingStrength, 0.0), 1.0)
        let minCutoff = 3.0 / (1.0 + s * 5.0)      // 3.0(轻) → 0.5(重)
        let beta = 0.015 + s * 0.02

        // 人数变化时同步滤波器数量
        if jointFilters.count > input.count {
            jointFilters.removeSubrange(input.count..<jointFilters.count)
        }
        while jointFilters.count < input.count {
            jointFilters.append((0..<Self.jointNames.count).map { _ in
                JointFilter(minCutoff: minCutoff, beta: beta)
            })
        }

        var out = input
        for (pi, pose) in input.enumerated() {
            var smoothed = pose
            var lost = true
            var minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0
            var any = false

            for slot in 0..<pose.points.count {
                guard let j = pose.points[slot] else { continue }
                lost = false
                let f = jointFilters[pi][slot]
                let nx = f.fx.filter(Double(j.position.x), at: time)
                let ny = f.fy.filter(Double(j.position.y), at: time)
                let np = CGPoint(x: nx, y: ny)
                smoothed.points[slot] = Joint(index: j.index, name: j.name,
                                              position: np, confidence: j.confidence)
                if j.confidence >= detectionConfidence {
                    any = true
                    minX = min(minX, nx)
                    maxX = max(maxX, nx)
                    minY = min(minY, ny)
                    maxY = max(maxY, ny)
                }
            }

            if lost {
                for f in jointFilters[pi] { f.reset() }
                continue
            }
            if any {
                smoothed.boundingBox = CGRect(x: minX, y: minY,
                                              width: max(0, maxX - minX),
                                              height: max(0, maxY - minY))
            }
            out[pi] = smoothed
        }
        return out
    }

    /// 平滑参数变了要清掉历史，否则会拿旧参数的状态继续算
    func resetSmoothing() {
        jointFilters.removeAll()
    }

    // MARK: 把人拼成 Pose

    private static func buildPose2D(_ obs: VNHumanBodyPoseObservation,
                                    index: Int,
                                    minConfidence: Float) -> Pose? {
        guard let recognized = try? obs.recognizedPoints(.all) else { return nil }

        var slots = [Joint?](repeating: nil, count: jointNames.count)
        var minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0
        var any = false
        var confSum: Float = 0
        var confCount = 0

        for (jointName, slot) in vision2DJoints {
            guard let p = recognized[jointName] else { continue }
            slots[slot] = Joint(index: slot, name: jointNames[slot],
                                position: p.location, confidence: p.confidence)
            confSum += p.confidence
            confCount += 1
            if p.confidence >= minConfidence {
                any = true
                minX = min(minX, Double(p.location.x))
                maxX = max(maxX, Double(p.location.x))
                minY = min(minY, Double(p.location.y))
                maxY = max(maxY, Double(p.location.y))
            }
        }

        guard any else { return nil }
        let box = CGRect(x: minX, y: minY, width: max(0, maxX - minX), height: max(0, maxY - minY))
        let avg = confCount > 0 ? confSum / Float(confCount) : 0
        return Pose(id: index, points: slots, confidence: avg, boundingBox: box)
    }

    /// 3D 姿态模型（iOS 17+）。官方说对遮挡/背影这类难视角鲁棒性明显更好。
    /// 用 pointInImage 取 2D 投影来画骨架。
    private static func buildPose3D(_ obs: VNHumanBodyPose3DObservation, index: Int) -> Pose? {
        var slots = [Joint?](repeating: nil, count: jointNames.count)
        var minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0
        var any = false
        let conf = obs.confidence

        for (jointName, slot) in vision3DJoints {
            guard let p = try? obs.pointInImage(jointName) else { continue }
            // pointInImage 返回 VNPoint（x/y 是 Double），转成 CGPoint
            let location = CGPoint(x: p.x, y: p.y)
            slots[slot] = Joint(index: slot, name: jointNames[slot],
                                position: location, confidence: conf)
            any = true
            minX = min(minX, Double(location.x))
            maxX = max(maxX, Double(location.x))
            minY = min(minY, Double(location.y))
            maxY = max(maxY, Double(location.y))
        }

        guard any else { return nil }
        let box = CGRect(x: minX, y: minY, width: max(0, maxX - minX), height: max(0, maxY - minY))
        return Pose(id: index, points: slots, confidence: conf, boundingBox: box)
    }
}

// MARK: - 采集回调

extension PoseEstimator: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        handle(pixelBuffer: pixelBuffer)
    }
}
