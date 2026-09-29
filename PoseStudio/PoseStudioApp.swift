import SwiftUI
import Combine

// MARK: - 应用入口

@main
struct PoseStudioApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
                .statusBarHidden(true)
        }
    }
}

// MARK: - 全局设置（每个叠加层元素独立开关）

final class Settings: ObservableObject {

    // ---- 显示元素开关（互不影响）----
    @Published var showSkeleton: Bool = true      // 骨架连线
    @Published var showKeypoints: Bool = true     // 关节圆点
    @Published var showBox: Bool = true           // 边界框
    @Published var showLabel: Bool = true         // 标签 + 置信度
    @Published var showIDs: Bool = true           // 人物编号
    @Published var showNames: Bool = false        // 关节名称
    @Published var showFacePoints: Bool = true    // 面部关键点（眼/耳）
    @Published var showHUD: Bool = true           // 帧率 / 人数

    // ---- 样式 ----
    @Published var groupColors: Bool = true       // 分区配色（头/躯干/四肢不同色）
    @Published var cornerBox: Bool = true         // 四角括号框
    @Published var dimBackground: Bool = false    // 压暗背景突出骨架
    @Published var hideLowConfidence: Bool = true // 隐藏低置信度关节
    /// 火柴人模式：头部画成一个圆圈，并隐藏眼/耳/鼻等面部关键点
    @Published var stickHead: Bool = false

    // ---- 参数 ----
    @Published var detectionConfidence: Double = 0.30
    @Published var jointConfidence: Double = 0.30
    @Published var lineWidth: Double = 3
    @Published var jointRadius: Double = 5
    @Published var maxPeople: Double = 1

    // ---- 摄像头 ----
    @Published var useFrontCamera: Bool = false

    /// 骨架坐标手动旋转兜底（0/90/180/270）。
    /// 正常情况下自动对齐就已经正确，这里只是万一设备方向判断有偏差时的补救。
    @Published var rotationOverride: Int = 0
}
