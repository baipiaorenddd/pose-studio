import SwiftUI
import Foundation

struct ContentView: View {

    @StateObject private var settings = Settings()
    @StateObject private var estimator = PoseEstimator()
    @State private var panelOpen = true

    private let accent = Color(red: 0.00, green: 0.88, blue: 0.54)
    private let rotations = [0, 90, 180, 270]

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            CameraPreview(session: estimator.session)
                .ignoresSafeArea()

            OverlayCanvas(poses: estimator.poses,
                          videoSize: estimator.videoSize,
                          mirrored: estimator.overlayMirrored,
                          settings: settings)

            VStack(spacing: 0) {
                if settings.showHUD && estimator.isRunning { hud }
                Spacer()
            }

            if !estimator.isRunning { startOverlay }

            VStack(spacing: 0) {
                Spacer()
                quickBar
                if panelOpen { panel }
            }
        }
        .onAppear { syncToEstimator() }
        .onChange(of: settings.detectionConfidence) { _, v in estimator.detectionConfidence = Float(v) }
        .onChange(of: settings.maxPeople) { _, v in estimator.maxPeople = Int(v) }
        .onChange(of: settings.use3DModel) { _, v in estimator.use3DModel = v }
        .onChange(of: settings.smoothingEnabled) { _, v in
            estimator.smoothingEnabled = v
            estimator.resetSmoothing()
        }
        .onChange(of: settings.smoothingStrength) { _, v in
            estimator.smoothingStrength = v
            estimator.resetSmoothing()
        }
        .onChange(of: settings.fpsLimit) { _, v in
            estimator.fpsLimit = v
            estimator.updateFrameRate(v)     // 真正去改采集设备帧率
        }
        .onChange(of: settings.useFrontCamera) { _, v in estimator.switchCamera(front: v) }
    }

    // MARK: - 顶部状态条

    private var hud: some View {
        HStack(spacing: 6) {
            chip("FPS", String(format: "%.1f", estimator.fps), true)
            chip("人数", "\(estimator.poses.count)", false)
            chip("耗时", String(format: "%.0f", estimator.inferenceMS) + "ms", false)
            chip("方向", estimator.orientationLabel, false)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
    }

    private func chip(_ title: String, _ value: String, _ highlight: Bool) -> some View {
        HStack(spacing: 3) {
            Text(title).foregroundColor(.white.opacity(0.6))
            Text(value).foregroundColor(highlight ? accent : .white).bold()
        }
        .font(.system(size: 10, design: .monospaced))
        .padding(.horizontal, 8).padding(.vertical, 5)
        .background(.ultraThinMaterial)
        .clipShape(Capsule())
    }

    // MARK: - 常驻快捷条

    private var quickBar: some View {
        VStack(spacing: 8) {

            // 置信度
            HStack(spacing: 10) {
                Text("置信度")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(width: 46, alignment: .leading)
                Slider(value: $settings.detectionConfidence, in: 0.10...0.90)
                    .tint(accent)
                Text(String(format: "%.2f", settings.detectionConfidence))
                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                    .foregroundColor(accent)
                    .frame(width: 42, alignment: .trailing)
            }

            // 摄像头（点按，不用滑动开关）
            HStack(spacing: 6) {
                Text("摄像头")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(width: 46, alignment: .leading)
                pillButton("后置", active: !estimator.useFrontCamera) {
                    settings.useFrontCamera = false
                }
                pillButton("前置", active: estimator.useFrontCamera) {
                    settings.useFrontCamera = true
                }
                Spacer(minLength: 0)
            }

            // 光学变焦：切换物理镜头
            if !estimator.availableLenses.isEmpty {
                HStack(spacing: 6) {
                    Text("倍率")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.white.opacity(0.85))
                        .frame(width: 46, alignment: .leading)
                    ForEach(estimator.availableLenses) { lens in
                        pillButton(lens.label, active: estimator.currentLensID == lens.id) {
                            estimator.switchLens(lens.id)
                        }
                    }
                    Spacer(minLength: 0)
                    Text("光学")
                        .font(.system(size: 10))
                        .foregroundColor(.white.opacity(0.4))
                }
            }

            // 旋转兜底 + 展开设置
            HStack(spacing: 6) {
                Text("旋转")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white.opacity(0.85))
                    .frame(width: 46, alignment: .leading)
                ForEach(rotations, id: \.self) { deg in
                    pillButton("\(deg)°", active: settings.rotationOverride == deg) {
                        settings.rotationOverride = deg
                    }
                }
                Spacer(minLength: 4)
                Button {
                    withAnimation(.easeInOut(duration: 0.2)) { panelOpen.toggle() }
                } label: {
                    Text(panelOpen ? "收起" : "更多设置")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.black)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Color.white.opacity(0.85))
                        .clipShape(Capsule())
                }
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal, 10)
        .padding(.bottom, panelOpen ? 6 : 12)
    }

    private func pillButton(_ title: String, active: Bool,
                            action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: active ? .bold : .regular))
                .foregroundColor(active ? .black : .white)
                .padding(.horizontal, 11).padding(.vertical, 6)
                .background(active ? accent : Color.white.opacity(0.12))
                .clipShape(Capsule())
        }
    }

    // MARK: - 启动界面

    private var startOverlay: some View {
        VStack(spacing: 14) {
            Text("Pose Studio")
                .font(.system(size: 26, weight: .bold))
                .foregroundColor(.white)
            Text("实时人体姿态识别\n骨架 / 边界框，逐项独立开关")
                .font(.system(size: 13))
                .multilineTextAlignment(.center)
                .foregroundColor(.white.opacity(0.6))

            Button {
                syncToEstimator()
                estimator.start(front: settings.useFrontCamera)
            } label: {
                Text("开启摄像头")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.black)
                    .frame(width: 220, height: 50)
                    .background(accent)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .padding(.top, 4)

            if estimator.permissionDenied {
                Text(estimator.statusText)
                    .font(.system(size: 12))
                    .foregroundColor(.orange)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(26)
        .background(Color.black.opacity(0.75))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .padding(24)
    }

    // MARK: - 详细设置面板

    private var panel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {

                group("数据源") {
                    HStack(spacing: 10) {
                        Button(estimator.isRunning ? "重新开始" : "开始") {
                            syncToEstimator()
                            estimator.start(front: settings.useFrontCamera)
                        }
                        .buttonStyle(PillButton(background: accent, foreground: .black))

                        Button("停止") { estimator.stop() }
                            .buttonStyle(PillButton(background: Color(white: 0.25), foreground: .white))
                    }
                }

                group("显示元素（独立开关）") {
                    toggleRow("骨架连线", $settings.showSkeleton)
                    toggleRow("边界框", $settings.showBox)
                    toggleRow("标签 + 置信度", $settings.showLabel)
                    toggleRow("人物编号 #N", $settings.showIDs)
                    toggleRow("关节名称", $settings.showNames)
                    toggleRow("面部关键点（眼/耳）", $settings.showFacePoints)
                }

                group("样式") {
                    toggleRow("火柴人头部（圆圈）", $settings.stickHead)
                    toggleRow("头部圆圈填充", $settings.headFill)
                    toggleRow("分区配色（关 = 白色骨架）", $settings.groupColors)
                    toggleRow("四角括号框", $settings.cornerBox)
                    toggleRow("压暗背景突出骨架", $settings.dimBackground)
                    toggleRow("隐藏低置信度关节", $settings.hideLowConfidence)
                    toggleRow("显示帧率 / 人数", $settings.showHUD)
                }

                group("识别") {
                    toggleRow("3D 姿态模型（更稳，稍慢）", $settings.use3DModel)
                    toggleRow("关节点平滑（消除抖动）", $settings.smoothingEnabled)
                    sliderRow("平滑强度", $settings.smoothingStrength, 0...1, "%.2f")
                    sliderRow("推理帧率上限", $settings.fpsLimit, 1...120, "%.0f")
                    sliderRow("检测置信度", $settings.detectionConfidence, 0.10...0.90, "%.2f")
                    sliderRow("关节可见度阈值", $settings.jointConfidence, 0.05...0.90, "%.2f")
                    sliderRow("最多人数", $settings.maxPeople, 1...4, "%.0f")
                }

                group("外观") {
                    sliderRow("线宽", $settings.lineWidth, 1...10, "%.0f")
                }

                Text("识别方向：\(estimator.orientationLabel)　·　程序自动试探并锁定")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.5))

                Text("状态：\(estimator.statusText)")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.4))
                    .padding(.bottom, 20)
            }
            .padding(16)
        }
        .frame(maxHeight: 340)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .padding(.horizontal, 10)
        .padding(.bottom, 10)
    }

    // MARK: - 小组件

    private func group<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(accent)
                .textCase(.uppercase)
            content()
        }
    }

    private func toggleRow(_ title: String, _ binding: Binding<Bool>) -> some View {
        Toggle(isOn: binding) {
            Text(title).font(.system(size: 14)).foregroundColor(.white)
        }
        .toggleStyle(SwitchToggleStyle(tint: accent))
        .padding(.vertical, 1)
    }

    private func sliderRow(_ title: String, _ value: Binding<Double>,
                           _ range: ClosedRange<Double>, _ format: String) -> some View {
        VStack(spacing: 2) {
            HStack {
                Text(title).font(.system(size: 13)).foregroundColor(.white)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                    .foregroundColor(accent)
            }
            Slider(value: value, in: range)
                .tint(accent)
        }
        .padding(.vertical, 1)
    }

    private func syncToEstimator() {
        estimator.detectionConfidence = Float(settings.detectionConfidence)
        estimator.maxPeople = Int(settings.maxPeople)
        estimator.fpsLimit = settings.fpsLimit
        estimator.use3DModel = settings.use3DModel
        estimator.smoothingEnabled = settings.smoothingEnabled
        estimator.smoothingStrength = settings.smoothingStrength
    }
}

// MARK: - 胶囊按钮样式

struct PillButton: ButtonStyle {
    let background: Color
    let foreground: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 14, weight: .semibold))
            .foregroundColor(foreground)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(background.opacity(configuration.isPressed ? 0.6 : 1))
            .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}
