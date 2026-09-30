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
    @Published var showBox: Bool = true           // 边界框
    @Published var showLabel: Bool = true         // 标签 + 置信度
    @Published var showIDs: Bool = true           // 人物编号
    @Published var showNames: Bool = false        // 关节名称
    @Published var showFacePoints: Bool = false   // 面部关键点（眼/耳）
    @Published var showHUD: Bool = true           // 帧率 / 人数

    // ---- 样式 ----
    /// 分区配色。关闭 = 纯白骨架（默认）
    @Published var groupColors: Bool = false
    @Published var cornerBox: Bool = true         // 四角括号框
    @Published var dimBackground: Bool = false    // 压暗背景突出骨架
    @Published var hideLowConfidence: Bool = true // 隐藏低置信度关节
    /// 火柴人模式：头部画成一个圆圈，并隐藏眼/耳/鼻等面部关键点
    @Published var stickHead: Bool = true
    /// 头部圆圈是否填充。默认关 = 纯线条、完全透明
    @Published var headFill: Bool = false

    // ---- 参数 ----
    @Published var detectionConfidence: Double = 0.30
    @Published var jointConfidence: Double = 0.30
    @Published var lineWidth: Double = 3
    @Published var maxPeople: Double = 1
    /// 推理帧率上限（1–120）。会真正改采集设备帧率，不只是软件丢帧。
    @Published var fpsLimit: Double = 30
    /// 使用 3D 姿态模型（iOS 17+）。对背影/遮挡的鲁棒性更好，但稍慢。
    @Published var use3DModel: Bool = false

    // ---- 平滑（消除关键点抖动）----
    @Published var smoothingEnabled: Bool = true
    /// 0 = 轻，1 = 重
    @Published var smoothingStrength: Double = 0.5

    // ---- 摄像头 ----
    @Published var useFrontCamera: Bool = false

    /// 骨架坐标手动旋转兜底（0/90/180/270）。正常情况下 0 就是对的。
    @Published var rotationOverride: Int = 0
}
