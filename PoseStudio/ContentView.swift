import SwiftUI
import Foundation

struct ContentView: View {

    @StateObject private var settings = Settings()
    @StateObject private var estimator = PoseEstimator()
    @State private var panelOpen = false

    private let accent = Color(red: 0.00, green: 0.88, blue: 0.54)

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            CameraPreview(session: estimator.session)
                .ignoresSafeArea()

            OverlayCanvas(poses: estimator.poses,
                          videoSize: estimator.videoSize,
                          settings: settings)

            VStack(spacing: 0) {
                if settings.showHUD && estimator.isRunning { hud }
                Spacer()
            }

            if !estimator.isRunning { startOverlay }

            VStack(spacing: 0) {
                Spacer()
                HStack {
                    Spacer()
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) { panelOpen.toggle() }
                    } label: {
                        Image(systemName: panelOpen ? "xmark" : "slider.horizontal.3")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(.black)
                            .frame(width: 46, height: 46)
                            .background(accent)
                            .clipShape(Circle())
                            .shadow(radius: 6)
                    }
                    .padding(.trailing, 16)
                    .padding(.bottom, panelOpen ? 8 : 24)
                }
                if panelOpen { panel.transition(.move(edge: .bottom)) }
            }
        }
        .onAppear { syncToEstimator() }
        .onChange(of: settings.detectionConfidence) { _, v in estimator.detectionConfidence = Float(v) }
        .onChange(of: settings.maxPeople) { _, v in estimator.maxPeople = Int(v) }
        .onChange(of: settings.useFrontCamera) { _, v in estimator.switchCamera(front: v) }
    }

    // MARK: - 顶部状态条

    private var hud: some View {
        HStack(spacing: 8) {
            chip("FPS", String(format: "%.1f", estimator.fps), true)
            chip("人数", "\(estimator.poses.count)", false)
            chip("耗时", String(format: "%.0f ms", estimator.inferenceMS), false)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
    }

    private func chip(_ title: String, _ value: String, _ highlight: Bool) -> some View {
        HStack(spacing: 4) {
            Text(title).foregroundColor(.white.opacity(0.65))
            Text(value).foregroundColor(highlight ? accent : .white).bold()
        }
        .font(.system(size: 12, design: .monospaced))
        .padding(.horizontal, 9).padding(.vertical, 5)
        .background(Color.black.opacity(0.55))
        .clipShape(Capsule())
    }

    // MARK: - 启动界面

    private var startOverlay: some View {
        VStack(spacing: 16) {
            Text("Pose Studio")
                .font(.system(size: 26, weight: .bold))
                .foregroundColor(.white)
            Text("实时人体姿态识别\n骨架 / 边界框 / 连线，逐项独立开关")
                .font(.system(size: 13))
                .multilineTextAlignment(.center)
                .foregroundColor(.white.opacity(0.6))

            Button {
                estimator.detectionConfidence = Float(settings.detectionConfidence)
                estimator.maxPeople = Int(settings.maxPeople)
                estimator.start(front: settings.useFrontCamera)
            } label: {
                Text("开启摄像头")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.black)
                    .frame(width: 220, height: 50)
                    .background(accent)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .padding(.top, 8)

            if estimator.permissionDenied {
                Text(estimator.statusText)
                    .font(.system(size: 12))
                    .foregroundColor(.orange)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 30)
            }
        }
        .padding(28)
        .background(Color.black.opacity(0.72))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .padding(24)
    }

    // MARK: - 控制面板

    private var panel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {

                group("数据源") {
                    Toggle(isOn: $settings.useFrontCamera) {
                        Text("使用前置摄像头").font(.system(size: 14))
                    }
                    .toggleStyle(SwitchToggleStyle(tint: accent))

                    HStack(spacing: 10) {
                        Button(estimator.isRunning ? "重新开始" : "开始") {
                            estimator.detectionConfidence = Float(settings.detectionConfidence)
                            estimator.maxPeople = Int(settings.maxPeople)
                            estimator.start(front: settings.useFrontCamera)
                        }
                        .buttonStyle(PillButton(background: accent, foreground: .black))

                        Button("停止") { estimator.stop() }
                            .buttonStyle(PillButton(background: Color(white: 0.25), foreground: .white))
                    }
                    .padding(.top, 4)
                }

                group("显示元素（独立开关）") {
                    toggleRow("骨架连线", $settings.showSkeleton)
                    toggleRow("关节圆点", $settings.showKeypoints)
                    toggleRow("边界框", $settings.showBox)
                    toggleRow("标签 + 置信度", $settings.showLabel)
                    toggleRow("人物编号 #N", $settings.showIDs)
                    toggleRow("关节名称", $settings.showNames)
                    toggleRow("面部关键点（眼/耳）", $settings.showFacePoints)
                }

                group("样式") {
                    toggleRow("分区配色", $settings.groupColors)
                    toggleRow("四角括号框", $settings.cornerBox)
                    toggleRow("压暗背景突出骨架", $settings.dimBackground)
                    toggleRow("隐藏低置信度关节", $settings.hideLowConfidence)
                    toggleRow("显示帧率 / 人数", $settings.showHUD)
                }

                group("参数") {
                    sliderRow("检测置信度", $settings.detectionConfidence, 0.1...0.9, "%.2f")
                    sliderRow("关节阈值", $settings.jointConfidence, 0.05...0.9, "%.2f")
                    sliderRow("线宽", $settings.lineWidth, 1...10, "%.0f")
                    sliderRow("关节点大小", $settings.jointRadius, 1...12, "%.0f")
                    sliderRow("最多人数", $settings.maxPeople, 1...4, "%.0f")
                }

                Text("运行状态：\(estimator.statusText)　·　Vision 神经引擎加速，全部在本机处理")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.4))
                    .padding(.bottom, 24)
            }
            .padding(16)
        }
        .frame(maxHeight: 420)
        .background(Color(red: 0.09, green: 0.09, blue: 0.11).opacity(0.97))
        .clipShape(RoundedRectangle(cornerRadius: 18))
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
