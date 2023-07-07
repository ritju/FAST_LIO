#include "rclcpp/rclcpp.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include <omp.h> 

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// 枚举类型：表示支持的雷达类型
enum LID_TYPE{AVIA = 1, VELO16, OUST64}; //{1, 2, 3}
//时间戳单位
enum TIME_UNIT{SEC = 0, MS = 1, US = 2, NS = 3};
//枚举类型：表示特征点的类型 {正常点,可能的平面点，确定的平面点，有跨越的边，边上的平面点，线段 ，无效点 程序中未使用}
enum Feature{Nor, Poss_Plane, Real_Plane, Edge_Jump, Edge_Plane, Wire, ZeroPoint};
//枚举类型：位置标识
enum Surround{Prev, Next};
//枚举类型：表示有跨越边的类型{正常，0，180，无穷大，盲区}
enum E_jump{Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind};

//orgtype类：用于存储激光雷达点的一些其他属性
struct orgtype
{
  double range; //点云在xy平面离雷达中心的距离
  double dista; //当前点与后一个点之间的距离
  //假设雷达原点为O 前一个点为M 当前点为A 后一个点为N
  double angle[2]; //这个是角OAM和角OAN的cos值
  double intersect; //这个是角MAN的cos值
  E_jump edj[2]; //前后两点的类型
  Feature ftype; //点类型

  //构造函数
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor; //默认为正常点
    intersect = 2;
  }
};

// velodyne数据结构
namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D; // 4D点坐标类型
      float intensity; // 强度
      float time; // 时间
      uint16_t ring; // 点所属的圈数,线数
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW //进行内存对齐
  };
}  // namespace velodyne_ros

//// 注册velodyne_ros的Point类型
POINT_CLOUD_REGISTER_POINT_STRUCT(
  velodyne_ros::Point,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (float, time, time)
  (uint16_t, ring, ring)
)

// ouster数据结构
namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D; // 4D点坐标类型
      float intensity; // 强度
      uint32_t t; //时间戳
      uint16_t reflectivity; //反射率
      uint8_t  ring; //线数
      uint16_t ambient; //没用到
      uint32_t range; //距离
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW //进行内存对齐
  };
}  // namespace ouster_ros

// clang-format off
// 注册ouster的Point类型
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();
  
  void process(const livox_ros_driver2::msg::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out); // 对Livox自定义Msg格式的激光雷达数据进行处理
  void process(const sensor_msgs::msg::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);// 对ros的Msg格式的激光雷达数据进行处理
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf; // 全部点、边缘点、平面点
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;// 雷达类型、采样间隔、扫描线数、扫描频率
  double blind; // 最小距离阈值(盲区)
  bool feature_enabled, given_offset_time; // 是否提取特征、是否进行时间偏移
  //ros::Publisher pub_full, pub_surf, pub_corn; //目前没有用
    

  private:
  void avia_handler(const livox_ros_driver2::msg::CustomMsg::ConstPtr &msg); // 用于对Livox激光雷达数据进行处理
  void oust64_handler(const sensor_msgs::msg::PointCloud2::ConstPtr &msg); // 用于对ouster激光雷达数据进行处理
  void velodyne_handler(const sensor_msgs::msg::PointCloud2::ConstPtr &msg); // 用于对velodyne激光雷达数据进行处理
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const rclcpp::Time &ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct); // 没有用到
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};
