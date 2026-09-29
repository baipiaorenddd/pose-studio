import Foundation
import AVFoundation
import Vision
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
    var points: [Joint?]        // 固定 19 项，索引对应 Self.joints 顺序
    var confidence: Float
    var boundingBox: CGRect     // Vision 归一化坐标（原点左下）
}

// MARK: - 姿态推理

final class PoseEstimator: NSObject, ObservableObject {

    // 19 个关节（Apple Vision 人体姿态）
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

    // 发布给界面的状态
    @Published var poses: [Pose] = []
    @Published var fps: Double = 0
    @Published var inferenceMS: Double = 0
    @Published var videoSize: CGSize = .zero
    @Published var isRunning = false
    @Published var permissionDenied = false
    @Published var statusText: String = "未启动"

    // 由界面同步进来的参数（避免跨线程读 ObservableObject）
    var detectionConfidence: Float = 0.30
    var maxPeople: Int = 1

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "posestudio.session")
    private let videoQueue = DispatchQueue(label: "posestudio.video")
    private let request = VNDetectHumanBodyPoseRequest()
    private var videoOutput: AVCaptureVideoDataOutput?
    private var currentInput: AVCaptureDeviceInput?
    private var useFront = false
    private var configured = false

    private var frameCount = 0
    private var lastFPSAt = CFAbsoluteTimeGetCurrent()

    private static let visionJoints: [(VNHumanBodyPoseObservation.JointName, Int)] = [
        (.nose, 0), (.leftEye, 1), (.rightEye, 2), (.leftEar, 3), (.rightEar, 4),
        (.neck, 5), (.leftShoulder, 6), (.rightShoulder, 7),
        (.leftElbow, 8), (.rightElbow, 9), (.leftWrist, 10), (.rightWrist, 11),
        (.leftHip, 12), (.rightHip, 13), (.leftKnee, 14), (.rightKnee, 15),
        (.leftAnkle, 16), (.rightAnkle, 17), (.root, 18),
    ]

    // MARK: 权限 + 启动

    func start(front: Bool) {
        useFront = front
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
        statusText = "启动中…"
        sessionQueue.async {
            self.configureIfNeeded()
            if !self.session.isRunning { self.session.startRunning() }
            DispatchQueue.main.async {
                self.isRunning = true
                self.statusText = self.useFront ? "前置摄像头" : "后置摄像头"
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

    /// 切换前后摄像头：重建输入
    func switchCamera(front: Bool) {
        useFront = front
        sessionQueue.async {
            guard self.configured else { return }
            self.session.beginConfiguration()
            if let old = self.currentInput {
                self.session.removeInput(old)
                self.currentInput = nil
            }
            let position: AVCaptureDevice.Position = front ? .front : .back
            if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
               let input = try? AVCaptureDeviceInput(device: device),
               self.session.canAddInput(input) {
                self.session.addInput(input)
                self.currentInput = input
            }
            self.applyConnectionSettings(position: position)
            self.session.commitConfiguration()
        }
    }

    // MARK: 会话配置

    private func configureIfNeeded() {
        guard !configured else { return }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        let position: AVCaptureDevice.Position = useFront ? .front : .back
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.statusText = "找不到可用摄像头" }
            return
        }
        session.addInput(input)
        currentInput = input

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

        applyConnectionSettings(position: position)
        session.commitConfiguration()
        configured = true
    }

    /// 让数据输出与预览层方向一致：竖屏旋转 90°，前置摄像头镜像
    private func applyConnectionSettings(position: AVCaptureDevice.Position) {
        guard let conn = videoOutput?.connection(with: .video) else { return }
        if conn.isVideoRotationAngleSupported(90) {
            conn.videoRotationAngle = 90
        }
        if conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = (position == .front)
        }
    }

    // MARK: 每帧推理

    fileprivate func handle(pixelBuffer: CVPixelBuffer) {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        let t0 = CFAbsoluteTimeGetCurrent()
        do {
            try handler.perform([request])
        } catch {
            return
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        var found: [Pose] = []
        if let observations = request.results {
            let limit = max(1, maxPeople)
            for (i, obs) in observations.prefix(limit).enumerated() {
                if let pose = Self.buildPose(obs, index: i, minConfidence: detectionConfidence) {
                    found.append(pose)
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

        DispatchQueue.main.async {
            self.poses = found
            self.inferenceMS = elapsed
            if self.videoSize.width != CGFloat(width) || self.videoSize.height != CGFloat(height) {
                self.videoSize = CGSize(width: width, height: height)
            }
            if newFPS > 0 { self.fps = newFPS }
        }
    }

    private static func buildPose(_ obs: VNHumanBodyPoseObservation,
                                  index: Int,
                                  minConfidence: Float) -> Pose? {
        guard let recognized = try? obs.recognizedPoints(.all) else { return nil }

        var slots = [Joint?](repeating: nil, count: jointNames.count)
        var minX = 1.0, minY = 1.0, maxX = 0.0, maxY = 0.0
        var any = false
        var confSum: Float = 0
        var confCount = 0

        for (jointName, slot) in visionJoints {
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
