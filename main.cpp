#include "geometrycentral/pointcloud/point_cloud_heat_solver.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/transfer_functions.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <polyscope/polyscope.h>
#include <polyscope/pick.h>  // ✅ 确保包含这个头文件.


#include "args/args.hxx"
#include "imgui.h"

#include "utils.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

#include <chrono>
using std::chrono::duration;    //表示一段时间间隔
using std::chrono::duration_cast;   //将时间单位转换为毫秒等格式
using std::chrono::high_resolution_clock;  //提供最高精度的计时器
using std::chrono::milliseconds;    //表示时间单位是 毫秒（std::milli）

// == Geometry-central data
//这些变量存储 网格或点云的几何数据，用于 计算 GSD
std::unique_ptr<SurfaceMesh> mesh;  //网格拓扑结构（包含顶点、边、面）
std::unique_ptr<VertexPositionGeometry> geometry;   //网格 顶点坐标、法向量、面积、角度等几何信息
std::unique_ptr<pointcloud::PointCloud> cloud;  //点云结构（如果输入数据是点云）
std::unique_ptr<pointcloud::PointPositionGeometry> pointGeom;   //点云的 几何信息，包括点坐标
pointcloud::PointData<Vector3> pointPositions, pointNormals;    
//pointPositions：点云 顶点坐标 pointNormals：点云 法向量

// Polyscope stuff
//可视化对象
polyscope::SurfaceMesh* psMesh; //存储网格的可视化对象
polyscope::PointCloud* psCloud; //存储点云的可视化对象


// MESH_MODE 代表当前数据的类型：
// Triangle（三角网格）
// Polygon（多边形网格）
// Points（点云）
enum MeshMode { Triangle = 0, Polygon, Points };
int MESH_MODE = MeshMode::Triangle;

// == Intrinsic triangulation stuff. All remeshing is performed on the manifold mesh.
//内在三角剖分
VertexData<Vector3> csPositions; //存储 网格顶点的变换坐标
std::unique_ptr<ManifoldSurfaceMesh> manifoldMesh; //流形网格（用于重建 Intrinsic Mesh）
std::unique_ptr<VertexPositionGeometry> manifoldGeom, csGeom; //流形网格的几何信息
std::unique_ptr<IntegerCoordinatesIntrinsicTriangulation> intTri; //内在三角剖分（Intrinsic Triangulation）数据结构

//存储 GSD 计算所需的约束点和曲线
std::vector<Curve> CURVES, curvesOnManifold, curvesOnIntrinsic; //约束曲线()
std::vector<SurfacePoint> POINTS, pointsOnManifold, pointsOnIntrinsic; //约束点
std::vector<std::vector<pointcloud::Point>> POINTS_CURVES;
std::vector<std::vector<Vertex>> POLYGON_CURVES;

//用于 Intrinsic Mesh 细分和优化
// REFINE_AREA_THRESH：网格细分阈值。
// REFINE_ANGLE_THRESH：网格重建角度阈值 ? ->有什么用?
// INTTRI_UPDATED：标记 Intrinsic Mesh 是否更新。
float REFINE_AREA_THRESH = std::numeric_limits<float>::infinity();
float REFINE_ANGLE_THRESH = 25.;
int MAX_INSERTIONS = -1;
bool INTTRI_UPDATED = false;

//SOLVER_MODE 决定 GSD 求解模式
// ExtrinsicMesh（外在网格计算）
// IntrinsicMesh（内在网格计算）
enum SolverMode { ExtrinsicMesh = 0, IntrinsicMesh };
int SOLVER_MODE = SolverMode::ExtrinsicMesh;
int LAST_SOLVER_MODE;

// Solvers & parameters
//求解器 & 计算参数
float TCOEF = 1.; //热扩散时间系数（影响扩散范围）
bool TIME_UPDATED = false; //标记时间参数是否更新
SignedHeatOptions SHM_OPTIONS; //SignedHeatMethod 选项
int CONSTRAINT_MODE = static_cast<int>(LevelSetConstraint::ZeroSet); //控制符号距离的边界条件
bool SOLVE_AS_POINT_CLOUD = false;
bool VIZ = true; 
bool VERBOSE, HEADLESS, IS_POLY; 

// GSD 求解器：
// signedHeatSolver：外在网格 GSD 计算。
// intrinsicSolver：内在网格 GSD 计算。
// pointCloudSolver：点云 GSD 计算
std::unique_ptr<SignedHeatSolver> signedHeatSolver, intrinsicSolver;
std::unique_ptr<pointcloud::PointCloudHeatSolver> pointCloudSolver;

// PHI：存储 GSD 结果。
// PHI_CS：内在网格上的 GSD 结果
VertexData<double> PHI, PHI_CS;
pointcloud::PointData<double> PHI_POINTS;

// Program variables

//网格文件路径、输出文件路径等信息
std::string MESHNAME = "input mesh";
std::string MESH_FILEPATH, OUTPUT_FILENAME, DATA_DIR;
std::string OUTPUT_DIR = "../export";

bool VIS_INTRINSIC_MESH = false;
bool COMMON_SUBDIVISION = true;
bool USE_BOUNDS = false;
float LOWER_BOUND, UPPER_BOUND;
bool CONSTRAINED_GEOM;
bool EXPORT_ON_CS = true;

//用于记录计算时间（用于性能分析）
std::chrono::time_point<high_resolution_clock> t1, t2;
std::chrono::duration<double, std::milli> ms_fp;

// ensureHaveIntrinsicSolver() 的功能
// 检查 intrinsicSolver 是否已创建（如果已有，直接返回）。
// 确保网格满足 Intrinsic Mesh 计算要求：
// 是流形网格（isManifold()）。
// 方向正确（isOriented()）。
// 受约束的几何结构（CONSTRAINED_GEOM）。
// 调用 setIntrinsicSolver() 创建 intrinsicSolver：
// 生成 Intrinsic Triangulation（intTri）。
// 在 Intrinsic Mesh 上创建 SignedHeatSolver。
// 记录 intrinsicSolver 构造时间，并输出日志（如果 VERBOSE 为 true）。
void ensureHaveIntrinsicSolver() {

// 确保 intrinsicSolver 已经被创建，用于 内在网格（Intrinsic Mesh） 计算 广义符号距离（GSD）
    if (intrinsicSolver != nullptr) return;

    //检查网格是否是流形（Manifold）
    //检查网格是否方向一致（Oriented）
    //表示约束几何是否存在
    if (mesh->isManifold() && mesh->isOriented() & CONSTRAINED_GEOM) {
        //进入求解器构造流程
        //开启详细日志(VERBOSE),输出提示信息
        if (VERBOSE) std::cerr << "Constructing intrinsic solver..." << std::endl;

        //记录开始时间
        t1 = high_resolution_clock::now();

        //调用 setIntrinsicSolver() 进行初始化，负责创建 intrinsicSolver
        setIntrinsicSolver(*geometry, CURVES, POINTS, manifoldMesh, manifoldGeom, curvesOnManifold, pointsOnManifold,
                           intTri, intrinsicSolver);

        //记录结束时间
        t2 = high_resolution_clock::now();

        //计算intrinsicSolver 构造时间(单位:ms)
        ms_fp = t2 - t1;

        //如果 VERBOSE = true，输出 Intrinsic solver construction time
        //ms_fp.count() / 1000. 转换成秒（默认是毫秒 std::milli）
        if (VERBOSE) std::cerr << "Intrinsic solver construction time (s): " << ms_fp.count() / 1000. << std::endl;
    }
}

//该函数用于 计算广义符号距离（GSD, Generalized Signed Distance）。根据 不同的网格类型（MESH_MODE）和求解模式（SOLVER_MODE），采用不同的计算方式
void solve() {

    //处理三角网格
    if (MESH_MODE == MeshMode::Triangle) {
        //外在网格求解
        if (SOLVER_MODE == SolverMode::ExtrinsicMesh) {

            //设置 GSD 计算的边界条件：
            //通过 CONSTRAINT_MODE 指定 零水平集（GSD=0 的曲线）
            SHM_OPTIONS.levelSetConstraint = static_cast<LevelSetConstraint>(CONSTRAINT_MODE);

            //更新热扩散时间系数 TCOEF（影响扩散范围）
            if (TIME_UPDATED) signedHeatSolver->setDiffusionTimeCoefficient(TCOEF);

            //记录计算时间
            t1 = high_resolution_clock::now();//开始时间t1
            PHI = signedHeatSolver->computeDistance(CURVES, POINTS, SHM_OPTIONS); //调用函数计算外在GSD,结果存储在PHI中,可在GUI中可视化
            t2 = high_resolution_clock::now(); //结束时间t2
            ms_fp = t2 - t1; //计算时间

            //开启日志记录信息
            if (VERBOSE) {
                std::cerr << "min: " << PHI.toVector().minCoeff() << "\tmax: " << PHI.toVector().maxCoeff()
                          << std::endl;
                std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;
            }

            if (!HEADLESS) {//启用GUI可视化
                if (SHM_OPTIONS.levelSetConstraint != LevelSetConstraint::Multiple) {
                    //将 PHI 结果添加到 Polyscope 进行可视化
                    //若约束模式 不是 Multiple，使用 颜色映射 显示 GSD。
                    psMesh->addVertexSignedDistanceQuantity("GSD", PHI)->setEnabled(true);
                } else {
                    //若 约束模式是 Multiple，则显示 等值线（Isolines）
                    // If there's multiple level sets, it's arbitrary which one should be "zero".
                    psMesh->addVertexScalarQuantity("GSD", PHI)->setIsolinesEnabled(true)->setEnabled(true);
                }
            }
            //记录最后使用的求解模式为外在网格（ExtrinsicMesh）
            LAST_SOLVER_MODE = SolverMode::ExtrinsicMesh;

        } 
        //内在求解(Intrinsic Mesh)
        else {
            //确保 intrinsicSolver 存在,若不存在则初始化
            ensureHaveIntrinsicSolver();

            //将输入的source曲线和点投影到Intrinsic Mesh
            determineSourceGeometryOnIntrinsicTriangulation(*intTri, curvesOnManifold, pointsOnManifold,
                                                            curvesOnIntrinsic, pointsOnIntrinsic);
            //如果 Intrinsic Mesh 更新，则重置 intrinsicSolver
            if (INTTRI_UPDATED) {
                intrinsicSolver.reset(new SignedHeatSolver(*intTri));
                INTTRI_UPDATED = false;
            }
            //通过 CONSTRAINT_MODE 指定 零水平集
            SHM_OPTIONS.levelSetConstraint = static_cast<LevelSetConstraint>(CONSTRAINT_MODE);
            if (TIME_UPDATED) intrinsicSolver->setDiffusionTimeCoefficient(TCOEF);

            // 在 Intrinsic Mesh 上计算 GSD：
            // PHI_CS 存储 Intrinsic Mesh 上的 GSD 结果
            t1 = high_resolution_clock::now();
            PHI_CS = intrinsicSolver->computeDistance(curvesOnIntrinsic, pointsOnIntrinsic, SHM_OPTIONS);
            t2 = high_resolution_clock::now();
            ms_fp = t2 - t1;
            // 如果 VERBOSE = true，输出：
            // PHI_CS 计算出的 GSD 最小值 & 最大值。
            // 求解所需时间。
            if (VERBOSE) {
                std::cerr << "min: " << PHI_CS.toVector().minCoeff() << "\tmax: " << PHI_CS.toVector().maxCoeff()
                          << std::endl;
                std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;
            }

            //将 Intrinsic Mesh 的 GSD 结果 PHI_CS 映射回 Extrinsic Mesh
            PHI = transferBtoA(*intTri, PHI_CS, TransferMethod::L2);

            //在 GUI 中可视化 Intrinsic Mesh 计算的 GSD 结果
            if (!HEADLESS) {
                if (SHM_OPTIONS.levelSetConstraint != LevelSetConstraint::Multiple) {
                    //单个零水平集
                    psMesh->addVertexSignedDistanceQuantity("GSD", PHI)->setEnabled(true);
                    visualizeOnCommonSubdivision(*intTri, *manifoldGeom, csPositions, csGeom, PHI_CS, "GSD", true,
                                                 true);
                } else {
                    //多个零水平集
                    // If there's multiple level sets, it's arbitrary which one should be "zero".
                    psMesh->addVertexScalarQuantity("GSD", PHI)->setIsolinesEnabled(true)->setEnabled(true);
                    visualizeOnCommonSubdivision(*intTri, *manifoldGeom, csPositions, csGeom, PHI_CS, "GSD", true,
                                                 false);
                }
            }
            //记录最后使用的求解模式 为 内在网格（IntrinsicMesh）
            LAST_SOLVER_MODE = SolverMode::IntrinsicMesh;
        }
    }
    //处理点云
    else if (MESH_MODE == MeshMode::Points) {
        SHM_OPTIONS.levelSetConstraint = static_cast<LevelSetConstraint>(CONSTRAINT_MODE);
        // Reset point cloud solver in case parameters changed.
        if (TIME_UPDATED) pointCloudSolver.reset(new pointcloud::PointCloudHeatSolver(*cloud, *pointGeom, TCOEF));

        t1 = high_resolution_clock::now();
        PHI_POINTS = pointCloudSolver->computeSignedDistance(POINTS_CURVES, pointNormals, SHM_OPTIONS);
        t2 = high_resolution_clock::now();
        ms_fp = t2 - t1;
        if (VERBOSE) {
            std::cerr << "min: " << PHI_POINTS.toVector().minCoeff() << "\tmax: " << PHI_POINTS.toVector().maxCoeff()
                      << std::endl;
            std::cerr << "Solve time (s): " << ms_fp.count() / 1000. << std::endl;
        }

        if (!HEADLESS) psCloud->addScalarQuantity("GSD", PHI_POINTS)->setIsolinesEnabled(true)->setEnabled(true);

    } 
    
    //多边形网格的 GSD 计算尚未实现，抛出异常
    else if (MESH_MODE == MeshMode::Polygon) {
        throw std::logic_error("SHM on polygon meshes is not yet released - ETA September-October 2024");
        // TODO: Set up polygon mesh solver
        // signedHeatSolver->setDiffusionTimeCoefficient(TCOEF);
        // Vector<double> phi = signedHeatSolver->solve(CURVES, POINTS, SHM_OPTIONS);

        // if (!HEADLESS) psMesh->addVertexSignedDistanceQuantity("GSD", phi)->setEnabled(true);
        // if (EXPORT_RESULT) {
        //     exportCurves(geometry->vertexPositions, CURVES, POINTS, OUTPUT_DIR + "/source.obj");
        //     exportSDF(*geometry, PHI, OUTPUT_FILENAME, USE_BOUNDS, LOWER_BOUND, UPPER_BOUND);
        // }
    }

    TIME_UPDATED = false;
}

// callback()是一个 GUI 交互回调函数，基于 ImGui 构建了 用户界面（UI），用于：
// 执行 GSD 计算（Solve）
// 导出计算结果
// 调整求解参数
// 切换 Extrinsic 和 Intrinsic 计算模式
// 优化 Intrinsic Mesh
void callback() {

    //创建一个按钮，点击后调用solve()计算GSD
    if (ImGui::Button("Solve")) {
        solve();
    }

    //创建一个按钮，用户点击后导出上一次 GSD 计算的结果
    if (ImGui::Button("Export last solution")) {
        if (MESH_MODE == MeshMode::Triangle) {
            //三角网格->导出输入的曲线 CURVES 和 POINTS，保存到 OUTPUT_DIR
            exportCurves(geometry->vertexPositions, CURVES, POINTS, OUTPUT_DIR);
            // 如果最近一次计算模式是 ExtrinsicMesh：
            // 调用 exportSDF() 导出 PHI（外在网格的 GSD 结果）
            if (LAST_SOLVER_MODE == SolverMode::ExtrinsicMesh) {
                exportSDF(*geometry, PHI, OUTPUT_FILENAME, USE_BOUNDS, LOWER_BOUND, UPPER_BOUND);
            } 

            //如果最近一次计算模式是 IntrinsicMesh：
            // 如果 EXPORT_ON_CS = false，则导出 转换后的 PHI（映射回 Extrinsic Mesh 的 GSD）。
            // 如果 EXPORT_ON_CS = true，则导出 Intrinsic Mesh 的 PHI_CS 结果
            else if (LAST_SOLVER_MODE == SolverMode::IntrinsicMesh) {
                if (!EXPORT_ON_CS) {
                    exportSDF(*geometry, PHI, OUTPUT_FILENAME, USE_BOUNDS, LOWER_BOUND, UPPER_BOUND);
                } else {
                    exportSDF(*intTri, *manifoldGeom, PHI_CS, OUTPUT_FILENAME, USE_BOUNDS, LOWER_BOUND, UPPER_BOUND);
                }
            }
        } 

        //处理点云，导出点云的curves和GSD
        else if (MESH_MODE == MeshMode::Points) {
            exportCurves(pointGeom->positions, POINTS_CURVES, OUTPUT_DIR + "/source.obj");
            exportSDF(pointGeom->positions, PHI_POINTS, OUTPUT_FILENAME);
        } 
        //处理多边形网格
        else if (MESH_MODE == MeshMode::Polygon) {
            // TODO
        }
    }

    //如果最近一次计算模式是 Intrinsic Mesh：
    //用户可以勾选 EXPORT_ON_CS，决定 导出 Intrinsic Mesh 结果还是 Extrinsic Mesh 结果。


    if (LAST_SOLVER_MODE == SolverMode::IntrinsicMesh)
        ImGui::Checkbox("On common subdivision (vs. input mesh)", &EXPORT_ON_CS);

    ImGui::Text("Solve options");
    ImGui::Separator();

    // 用户可以输入 TCOEF（热扩散时间系数），影响 GSD 计算。
    // 如果 TCOEF 被修改，则 TIME_UPDATED = true，强制更新求解器
    if (ImGui::InputFloat("tCoef", &TCOEF)) TIME_UPDATED = true;

    //约束条件
    //是否保持源法向量（对曲面 GSD 计算有影响）
    ImGui::Checkbox("Preserve source normals", &SHM_OPTIONS.preserveSourceNormals);

    //用户可以选择零水平集的约束方式
    //单一零水平集(RadioButton->单选)
    ImGui::RadioButton("Constrain zero set", &CONSTRAINT_MODE, static_cast<int>(LevelSetConstraint::ZeroSet));
    //多个零水平集
    ImGui::RadioButton("Constrain multiple levelsets", &CONSTRAINT_MODE,
                       static_cast<int>(LevelSetConstraint::Multiple));
    //不使用零水平集
    ImGui::RadioButton("No levelset constraints", &CONSTRAINT_MODE, static_cast<int>(LevelSetConstraint::None));

    //软权重->调整 softLevelSetWeight（控制 SDF 计算中的平滑程度）
    ImGui::InputDouble("soft weight", &(SHM_OPTIONS.softLevelSetWeight));

    //设定导出范围
    //用户可以设置 USE_BOUNDS，手动指定导出的 GSD 范围
    ImGui::Checkbox("Specify upper/lower bounds for export", &USE_BOUNDS);
    if (ImGui::TreeNode("Bounds")) {
        ImGui::InputFloat("lower", &LOWER_BOUND);
        ImGui::InputFloat("upper", &UPPER_BOUND);
        ImGui::TreePop();
    }

    //     ImGui::TreePop();
    // }

    //如果网格是三角网格 & 是流形 & 受约束，则允许用户选择计算模式：
    if (MESH_MODE == MeshMode::Triangle && mesh->isManifold() && CONSTRAINED_GEOM) {
        ImGui::RadioButton("Solve on extrinsic mesh", &SOLVER_MODE, SolverMode::ExtrinsicMesh);
        ImGui::RadioButton("Solve on intrinsic mesh", &SOLVER_MODE, SolverMode::IntrinsicMesh);

    //展开 "Intrinsic mesh improvement" 选项，用户可以优化 Intrinsic Mesh
        //显示 Intrinsic Mesh
        if (ImGui::TreeNode("Intrinsic mesh improvement")) {
            if (ImGui::Checkbox("Show intrinsic edges", &VIS_INTRINSIC_MESH)) {
                ensureHaveIntrinsicSolver();
                visualizeIntrinsicEdges(*intTri, *manifoldGeom, VIS_INTRINSIC_MESH);
                INTTRI_UPDATED = true;
            }
        
        //切换 Delaunay 三角形
            if (ImGui::Button("Flip to Delaunay")) {
                ensureHaveIntrinsicSolver();
                intTri->flipToDelaunay();
                VIS_INTRINSIC_MESH = true;
                visualizeIntrinsicEdges(*intTri, *manifoldGeom, VIS_INTRINSIC_MESH);
                INTTRI_UPDATED = true;
            }
        
            ImGui::InputFloat("Angle thresh", &REFINE_ANGLE_THRESH);
            ImGui::InputFloat("Area thresh", &REFINE_AREA_THRESH);
            ImGui::InputInt("Max insert", &MAX_INSERTIONS);

            //创建 ImGui 按钮，点击后触发 Delaunay 细分优化
            if (ImGui::Button("Delaunay refine")) {
                ensureHaveIntrinsicSolver();

                //调用 delaunayRefine()，对 Intrinsic Mesh 进行 Delaunay 细分：
                // 细分过小角度的三角形（防止瘦长三角形，提高网格质量）。
                // 细分过大的三角形（保证网格均匀分布）。
                // 控制细分过程中最多插入 MAX_INSERTIONS 个点。
                intTri->delaunayRefine(REFINE_ANGLE_THRESH, REFINE_AREA_THRESH, MAX_INSERTIONS);
                VIS_INTRINSIC_MESH = true;

                //启用 Intrinsic Mesh 可视化：
                // VIS_INTRINSIC_MESH = true 表示 要显示 Intrinsic Mesh 的边。
                // visualizeIntrinsicEdges() 渲染 Intrinsic Mesh。
                visualizeIntrinsicEdges(*intTri, *manifoldGeom, VIS_INTRINSIC_MESH);
                INTTRI_UPDATED = true;
            }
            
            //关闭 ImGui 折叠菜单，让 UI 结构清晰。
            ImGui::TreePop();
        }
    }

    //添加查询功能
    // if (polyscope::pick::haveSelection()) {
    // auto result = polyscope::pick::getSelection();

    // if (result.first != nullptr) {  // 检查是否选中了有效的对象
    //     std::cout << "成功选择了一个对象！" << std::endl;
    //     std::cout << "对象指针: " << result.first << std::endl;
    //     std::cout << "选中的索引: " << result.second << std::endl;
    // } else {
    //     std::cout << " 没有选中任何对象。" << std::endl;
    // }
    // }

    
}


int main(int argc, char** argv) {

    // Configure the argument parser
    //创建一个命令行参数解析器 parser，说明该程序的功能是“求解广义符号距离”
    args::ArgumentParser parser("Solve for generalized signed distance.");
    //添加 --help 选项，当用户请求帮助时，程序会显示帮助信息并退出
    args::HelpFlag help(parser, "help", "Display this help menu", {"help"});
    //定义一个必选参数 meshFilename，用于指定网格文件的路径。这里是位置参数（positional）
    args::Positional<std::string> meshFilename(parser, "mesh", "A mesh file.");
    //定义一个可选参数 inputFilename（或 -i/--input），用于指定输入曲线文件的路径（用作约束条件）
    args::ValueFlag<std::string> inputFilename(parser, "input", "Input curve filepath", {"i", "input"});
    //定义一个可选参数 outputFilename（或 -o/--output），用于指定输出文件名，程序将结果保存到此文件中
    args::ValueFlag<std::string> outputFilename(parser, "output", "Output filename", {"o", "output"});

    // 定义参数组 group，包含两个标志：
    // verbose（-V/--verbose）：启用详细输出，调试时输出更多信息。
    // headless（-h/--headless）：无图形界面模式，程序运行时不启动 GUI，而直接计算结果。
    args::Group group(parser);
    args::Flag verbose(group, "verbose", "Verbose output", {"V", "verbose"});
    args::Flag headless(group, "headless", "Don't use the GUI.", {"h", "headless"});

    // Parse args
    //尝试解析命令行参数：
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help&) {
        //如果用户请求帮助（args::Help 异常），打印帮助信息并退出
        std::cout << parser;
        return 0;
    } catch (args::ParseError& e) {
        //如果参数解析失败（args::ParseError 异常），输出错误信息和帮助内容，然后退出
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    //检查是否提供了网格文件路径，如果没有，则输出错误提示并退出程序
    if (!meshFilename) {
        std::cerr << "Please specify a mesh file as argument." << std::endl;
        return EXIT_FAILURE;
    }

    // Load mesh
    //从解析器中获取用户指定的网格文件路径，并赋值给全局变量 MESH_FILEPATH
    MESH_FILEPATH = args::get(meshFilename);
    //调用 getHomeDirectory() 函数，提取文件所在的目录，存入全局变量 DATA_DIR
    DATA_DIR = getHomeDirectory(MESH_FILEPATH);

    //根据文件扩展名判断数据类型
    std::string ext = MESH_FILEPATH.substr(MESH_FILEPATH.find_last_of(".") + 1);
    //如果扩展名为 "pc"，说明文件是点云数据，设置 MESH_MODE 为 MeshMode::Points；否则，默认认为是网格数据（通常为三角网格）
    MESH_MODE = (ext == "pc") ? MeshMode::Points : MeshMode::Triangle;

    //如果指定了输出文件，则将输出文件所在目录赋值给 OUTPUT_DIR
    if (outputFilename) OUTPUT_DIR = getHomeDirectory(args::get(outputFilename));
    //构造最终输出文件名
    OUTPUT_FILENAME = OUTPUT_DIR + "/GSD.obj";
    //将命令行标志 headless 和 verbose 的值分别存储到全局变量 HEADLESS 与 VERBOSE 中，用于后续判断是否启用 GUI 或详细输出
    HEADLESS = headless;
    VERBOSE = verbose;

    //处理点云数据
    if (MESH_MODE == MeshMode::Points) {
        //定义两个向量 positions 和 normals，分别存储点云的坐标和法向量。调用 readPointCloud() 读取点云数据
        std::vector<Vector3> positions, normals;
        std::tie(positions, normals) = readPointCloud(MESH_FILEPATH);

        //根据点的数量创建 PointCloud 对象，并初始化 pointPositions 与 pointNormals 数据结构，这里使用智能指针管理内存
        size_t nPts = positions.size();
        cloud = std::unique_ptr<pointcloud::PointCloud>(new pointcloud::PointCloud(nPts));
        pointPositions = pointcloud::PointData<Vector3>(*cloud);
        pointNormals = pointcloud::PointData<Vector3>(*cloud);

        //将从文件中读取的坐标和法向量逐个赋值到 pointPositions 与 pointNormals
        for (size_t i = 0; i < nPts; i++) {
            pointPositions[i] = positions[i];
            pointNormals[i] = normals[i];
        }

        //根据点云数据和点坐标构建 PointPositionGeometry 对象，包含点云的几何信息
        pointGeom = std::unique_ptr<pointcloud::PointPositionGeometry>(
            new pointcloud::PointPositionGeometry(*cloud, pointPositions));
        
        //创建 PointCloudHeatSolver 对象，用于后续点云上 GSD 的计算。TCOEF 是热扩散时间系数
        pointCloudSolver.reset(new pointcloud::PointCloudHeatSolver(*cloud, *pointGeom, TCOEF));
    } 
    //处理网格数据
    else {
        //读取网格数据，调用 readSurfaceMesh() 返回：
        //mesh：网格的拓扑信息（顶点、边、面）。
        //geometry：网格的几何信息（顶点坐标等）
        //std::tie->创建一个元组的左值引用,或者将一个元组解包为独立的对象
        std::tie(mesh, geometry) = readSurfaceMesh(MESH_FILEPATH);

        //这段注释掉的代码显示了如何对网格进行居中和平移缩放，以便统一数值范围，但目前未启用

        // // Center and scale.
        // Vector3 bboxMin, bboxMax;
        // std::tie(bboxMin, bboxMax) = boundingBox(*geometry);
        // double diag = (bboxMin - bboxMax).norm();
        // Vector3 center = centroid(*geometry);
        // for (Vertex v : mesh->vertices()) {
        //     Vector3 p = geometry->vertexPositions[v];
        //     geometry->vertexPositions[v] = (p - center) / diag;
        // }
        // geometry->refreshQuantities();

        //检查网格是否为三角形网格。如果不是，则将 MESH_MODE 改为 MeshMode::Polygon
        if (!mesh->isTriangular()) MESH_MODE = MeshMode::Polygon;

        //创建 SignedHeatSolver 对象，用于在网格上计算 GSD（热扩散 + 求解泊松方程）
        signedHeatSolver = std::unique_ptr<SignedHeatSolver>(new SignedHeatSolver(*geometry));
    }

    // Load source geometry.
    //加载输入约束几何（曲线/点）
    if (inputFilename) {
        //如果用户指定了输入曲线文件，读取该文件路径
        std::string filename = args::get(inputFilename);
        //根据 MESH_MODE 的不同，调用不同函数读取输入约束
        switch (MESH_MODE) {
            case (MeshMode::Triangle): {
                std::tie(CURVES, POINTS) = readInput(*mesh, filename);
                CONSTRAINED_GEOM = isSourceGeometryConstrained(CURVES, POINTS);
                break;
            }
            case (MeshMode::Polygon): {
                POLYGON_CURVES = readCurveVertices(*mesh, filename);
                break;
            }
            case (MeshMode::Points): {
                POINTS_CURVES = readCurvePoints(*cloud, filename);
                break;
            }
        }
    }

    // Visualize data.
    //如果不处于无界面模式，则启动 GUI 可视化
    if (!HEADLESS) {
        //初始化 Polyscope（可视化库），并将 callback() 函数设置为用户交互的回调
        polyscope::init();
        polyscope::state::userCallback = callback;

        //修改代码支持拾取功能
        //polyscope::pick::initializePick();               // 初始化拾取功能
        //polyscope::pick::setPointPickEnabled(true);      // 启用点拾取功能


        //根据数据类型注册对应的可视化对象
        switch (MESH_MODE) {
            case (MeshMode::Triangle): {
                //调用 registerSurfaceMesh() 注册网格，并调用 displayInput() 显示输入约束（曲线和点）
                psMesh = polyscope::registerSurfaceMesh(MESHNAME, geometry->vertexPositions, mesh->getFaceVertexList());
                psMesh->setAllPermutations(polyscopePermutations(*mesh));
                displayInput(geometry->vertexPositions, CURVES, POINTS);
                break;
            }
            case (MeshMode::Polygon): {
                //类似于三角网格，但显示的曲线数据不同
                psMesh = polyscope::registerSurfaceMesh(MESHNAME, geometry->vertexPositions, mesh->getFaceVertexList());
                displayInput(geometry->vertexPositions, POLYGON_CURVES);
                break;
            }
            case (MeshMode::Points): {
                //调用 registerPointCloud() 注册点云，并显示输入约束
                psCloud = polyscope::registerPointCloud("point cloud", pointPositions);
                displayInput(pointGeom->positions, POINTS_CURVES);
                break;
            }
        }
        //启动 Polyscope 显示 GUI 界面，让用户交互
        polyscope::show();
    } else {
        //如果处于无界面模式（HEADLESS 为真），则直接调用 solve() 计算 GSD，不启动 GUI
        solve();
    }

    return EXIT_SUCCESS;
}