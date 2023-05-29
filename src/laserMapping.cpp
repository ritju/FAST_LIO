// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#define BOOST_BIND_NO_PLACEHOLDERS

#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include "so3_math.h"
#include "rclcpp/rclcpp.hpp"
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/transform_datatypes.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Quaternion.h"
#include "geometry_msgs/msg/vector3.hpp"
#include "livox_ros_driver/msg/custom_msg.hpp"
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

using std::placeholders::_1;


#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer; //互斥锁
condition_variable sig_buffer; // 条件变量

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
// scan_count：接收到的激光雷达Msg的总数，publish_count：接收到的IMU的Msg的总数
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
// lidar_pushed：用于判断激光雷达数据是否从缓存队列中拿到meas中
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
static vector<double>       extrinT(3, 0.0);
static vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer; // 激光雷达数据时间戳缓存队列
deque<PointCloudXYZI::Ptr>        lidar_buffer; // 激光雷达数据缓存队列
deque<sensor_msgs::msg::Imu::ConstPtr> imu_buffer; // IMU数据缓存队列

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path; //包含了一系列位姿
nav_msgs::msg::Odometry odomAftMapped; //只包含了一个位姿
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess()); // 定义指向激光雷达数据的预处理类Preprocess的智能指针
shared_ptr<ImuProcess> p_imu(new ImuProcess()); // 定义指向IMU数据预处理类ImuProcess的智能指针
//唤醒所有线程
void SigHandle(int sig)
{
    flg_exit = true;
    // ROS_WARN("catch sig %d", sig);
    cout << "catch sig %d" << sig << endl;
    sig_buffer.notify_all(); //  会唤醒所有等待队列中阻塞的线程 线程被唤醒后，会通过轮询方式获得锁，获得锁前也一直处理运行状态，不会被再次阻塞。
}
//打印状态
inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}
//把点从body系转到world系
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    //下面式子最里面的括号是从雷达到IMU坐标系 然后从IMU转到世界坐标系
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

//把点从body系转到world系 和上面其实差不多
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}
//把点从body系转到world系 和上面其实大同小异
template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}
//把点从body系转到world系
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}
//把点从Lidar系转到IMU系
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}
//得到被剔除的点
void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);//返回被剔除的点
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    //若未初始化 以当前雷达点为中心 长宽高200*200*200的局部地图
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){ //vertex 顶点
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    //当前雷达系中心到各个地图边缘的距离
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        //如果距离边缘过近 则需要进行地图的挪动
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return; //不需要挪动就直接退回了
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

builtin_interfaces::msg::Time time_from_sec(double time)
{
    builtin_interfaces::msg::Time return_stamp;
    int sec;

    sec = (int)time;

    return_stamp.set__sec(sec);
    return_stamp.set__nanosec((time - sec)*1000000000);

    return return_stamp;
}
//下面的时间同步默认为false 而且同步策略没太看懂
//除了AVIA类型之外的雷达点云预处理
void standard_pcl_cbk(sensor_msgs::msg::PointCloud2::ConstPtr msg)
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (tf2_ros::timeToSec(msg->header.stamp) < last_timestamp_lidar)
    {
        // ROS_ERROR("lidar loop back, clear buffer");
        cout << "lidar loop back, clear buffer" << endl;
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);//点云预处理
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(tf2_ros::timeToSec(msg->header.stamp));
    last_timestamp_lidar = tf2_ros::timeToSec(msg->header.stamp);
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;//预处理时间
    mtx_buffer.unlock();
    sig_buffer.notify_all();// 唤醒所有线程
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false; // 时间同步flag，false表示未进行时间同步，true表示已经进行过时间同步/**
 /* @brief 订阅器sub_pcl的回调函数：接收Livox激光雷达的点云数据，对点云数据进行预处理（特征提取、降采样、滤波），并将处理后的数据保存到激光雷达数据队列中
 * 
 * @param msg Livox自定义的msg格式，包含Livox激光雷达点云数据
 * @return void
 */
void livox_pcl_cbk(livox_ros_driver::msg::CustomMsg::ConstPtr msg)
{
    // 互斥锁
    mtx_buffer.lock();
    // 点云预处理的开始时间
    double preprocess_start_time = omp_get_wtime();
    // 激光雷达扫描的总次数
    scan_count ++;
    // 如果当前扫描的激光雷达数据的时间戳比上一次扫描的激光雷达数据的时间戳早，需要将激光雷达数据缓存队列清空
    if (tf2_ros::timeToSec(msg->header.stamp) < last_timestamp_lidar)
    {
        // ROS_ERROR("lidar loop back, clear buffer");
        cout << "lidar loop back, clear buffer" << endl;
        lidar_buffer.clear();
    }
    last_timestamp_lidar = tf2_ros::timeToSec(msg->header.stamp);
    // 如果不需要进行时间同步，而imu时间戳和雷达时间戳相差大于10s，则输出错误信息
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }
    // time_sync_en为true时，当imu时间戳和雷达时间戳相差大于1s时，进行时间同步
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }
    // 用pcl点云格式保存接收到的激光雷达数据
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    // 对激光雷达数据进行预处理（特征提取或者降采样），其中p_pre是Preprocess类的智能指针
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    // 点云预处理的总时间
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();// 唤醒所有线程
}

/**
 * @brief 订阅器sub_imu的回调函数：接收IMU数据，将IMU数据保存到IMU数据缓存队列中
 * 
 * @param msg_in IMU Msg
 * @return void
 */
void imu_cbk(sensor_msgs::msg::Imu::ConstPtr msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->header.stamp = time_from_sec(tf2_ros::timeToSec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    // 将IMU和激光雷达点云的时间戳对齐
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        time_from_sec(timediff_lidar_wrt_imu + tf2_ros::timeToSec(msg_in->header.stamp));
    }

    double timestamp = tf2_ros::timeToSec(msg->header.stamp);

    mtx_buffer.lock();// 互斥锁
    // 如果当前IMU的时间戳小于上一个时刻IMU的时间戳，则IMU数据有误，将IMU数据缓存队列清空
    if (timestamp < last_timestamp_imu)
    {
        // ROS_WARN("imu loop back, clear buffer");
        cout << "imu loop back, clear buffer" << endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;
    // 将当前的IMU数据保存到IMU数据缓存队列中
    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();// 唤醒所有线程
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
/**
 * @brief 将激光雷达点云数据和IMU数据从缓存队列中取出，进行时间对齐，并保存到meas中
 * 
 * @param meas  用于保存当前正在处理的激光雷达数据和IMU数据
 * @return true 
 * @return false 
 */
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    //如果还没有把雷达数据放到meas中的话，就执行一下操作
    if(!lidar_pushed)
    {
        // 从激光雷达点云缓存队列中取出点云数据，放到meas中
        meas.lidar = lidar_buffer.front();
        // 记录该次雷达扫描的开始时间
        meas.lidar_beg_time = time_buffer.front();
        // 计算该次雷达扫描的结束时间
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            // ROS_WARN("Too few input point cloud!\n");
            cout << "Too few input point cloud!\n" << endl;
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) //如果某次扫描时间太短了
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;//就强行给end时刻赋值
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); //此次雷达点云结束时刻
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;//每次扫描需要的平均时间
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time) //最新的IMU时间戳(也就是队尾的)不能早于雷达的end时间戳
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    // 从IMU缓存队列中取出IMU数据，放到meas中
    double imu_time = tf2_ros::timeToSec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    // 拿出lidar_beg_time到lidar_end_time之间的所有IMU数据
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = tf2_ros::timeToSec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

void publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = time_from_sec(lidar_end_time);
        // laserCloudmsg.header.frame_id = "camera_init";
        laserCloudmsg.header.frame_id = "laser_link";
        pubLaserCloudFull->publish(laserCloudmsg);
        // pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}
//把去畸变的雷达系下的点云转到IMU系
void publish_frame_body(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = time_from_sec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    // pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = time_from_sec(lidar_end_time);
    // laserCloudFullRes3.header.frame_id = "camera_init";
    laserCloudFullRes3.header.frame_id = "laser_link";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
    // pubLaserCloudEffect.publish(laserCloudFullRes3);
}
//发布地图
void publish_map(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pubLaserCloudMap)
{
    sensor_msgs::msg::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = time_from_sec(lidar_end_time);
    //laserCloudMap.header.frame_id = "camera_init";
    laserCloudMap.header.frame_id = "laser_link";
    pubLaserCloudMap->publish(laserCloudMap);
    // pubLaserCloudMap.publish(laserCloudMap);
}
//设置输出的位姿
template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}
//发布里程计 发布tf变换
void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr & pubOdomAftMapped)
{
    // odomAftMapped.header.frame_id = "camera_init";
    // odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.frame_id = "odom";
    odomAftMapped.child_frame_id = "base_footprint";
    odomAftMapped.header.stamp = time_from_sec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    // pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        //设置协方差 P里面先是旋转后是位置 这个POSE里面先是位置后是旋转 所以对应的协方差要对调一下
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    tf2::Transform                   transform;
    tf2::Quaternion                  q;
    transform.setOrigin(tf2::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = odomAftMapped.header.stamp;
    // tf_msg.header.frame_id = "camera_init";
    // tf_msg.child_frame_id = "body";
    tf_msg.header.frame_id = "laser_link";
    tf_msg.child_frame_id = "base_link";
    tf_msg.transform = tf2::toMsg(transform);

    tf_broadcaster->sendTransform(tf_msg);
    // tf_broadcaster_->sendTransform(tf2_ros::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}
//每隔10个发布一下位姿
void publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = time_from_sec(lidar_end_time);
    // msg_body_pose.header.frame_id = "camera_init";
    msg_body_pose.header.frame_id = "laser_link";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
        // pubPath.publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))//找平面点的阈值
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        // RCLCPP_WARN("No Effective Points! \n");
        cout << "No Effective Points! \n" << endl;
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}
//FAST_LIO2主函数
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("laserMapping_node");
    // 从参数服务器读取参数值赋给变量（包括launch文件和launch读取的yaml文件中的参数）
    //是否发布路径的topic
    if(!node->get_parameter("publish/path_en", path_en))
        path_en = true;
    //是否发布当前正在扫描的点云的topic
    if(!node->get_parameter("publish/scan_publish_en", scan_pub_en))
        scan_pub_en = true;
    //是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，
    if(!node->get_parameter("publish/dense_publish_en", dense_pub_en))
        dense_pub_en = true;
    //是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    if(!node->get_parameter("publish/scan_bodyframe_pub_en", scan_body_pub_en))
        scan_body_pub_en = true;
    //卡尔曼滤波的最大迭代次数
    if(!node->get_parameter("max_iteration", NUM_MAX_ITERATIONS))
        NUM_MAX_ITERATIONS = 4;
    //地图保存路径
    if(!node->get_parameter("map_file_path", map_file_path))
        map_file_path = "";
    //激光雷达点云topic名称
    if(!node->get_parameter("common/lid_topic", lid_topic))
        lid_topic = "/livox/lidar";
    //IMU的topic名称
    if(!node->get_parameter("common/imu_topic", imu_topic))
        imu_topic = "/livox/imu";
    //是否需要时间同步，只有当外部未进行时间同步时设为true
    if(!node->get_parameter("common/time_sync_en", time_sync_en))
        time_sync_en = false;
    //
    if(!node->get_parameter("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu))
        time_diff_lidar_to_imu = 0.0;
    //VoxelGrid降采样时的体素大小
    if(!node->get_parameter("filter_size_corner", filter_size_corner_min))
        filter_size_corner_min = 0.5;
    //VoxelGrid降采样时的体素大小
    if(!node->get_parameter("filter_size_surf", filter_size_surf_min))
        filter_size_surf_min = 0.5;
    //VoxelGrid降采样时的体素大小
    if(!node->get_parameter("filter_size_map", filter_size_map_min))
        filter_size_map_min = 0.5;
    //地图的局部区域的长度（FastLio2论文中有解释）
    if(!node->get_parameter("cube_side_length", cube_len))
        cube_len = 200;
    //激光雷达的最大探测范围
    if(!node->get_parameter("mapping/det_range", DET_RANGE))
        DET_RANGE = 30.f;
    //激光雷达的视场角
    if(!node->get_parameter("mapping/fov_degree", fov_deg))
        fov_deg = 270.0;
    //IMU陀螺仪的协方差
    if(!node->get_parameter("mapping/gyr_cov", gyr_cov))
        gyr_cov = 0.1;
    //IMU加速度的协方差
    if(!node->get_parameter("mapping/acc_cov", acc_cov))
        acc_cov = 0.1;
    //IMU陀螺仪偏置的协方差
    if(!node->get_parameter("mapping/b_gyr_cov", b_gyr_cov))
        b_gyr_cov = 0.0001;
    //IMU加速度计偏置的协方差
    if(!node->get_parameter("mapping/b_acc_cov", b_acc_cov))
        b_acc_cov = 0.0001;
    //最小距离阈值，即过滤掉0～blind范围内的点云
    if(!node->get_parameter("preprocess/blind", p_pre->blind)) 
        p_pre->blind = 0.1;
    //激光雷达的类型
    if(!node->get_parameter("preprocess/lidar_type", p_pre->lidar_type))
        p_pre->lidar_type = VELO16;
    //激光雷达扫描的线数（livox avia为6线）
    if(!node->get_parameter("preprocess/scan_line", p_pre->N_SCANS))
        p_pre->N_SCANS = 1;
    //点云时间搓的单位
    if(!node->get_parameter("preprocess/timestamp_unit", p_pre->time_unit))
        p_pre->time_unit = MS;
    //雷达话题的频率
    if(!node->get_parameter("preprocess/scan_rate", p_pre->SCAN_RATE))
        p_pre->SCAN_RATE = 28;
    //采样间隔，即每隔point_filter_num个点取1个点
    if(!node->get_parameter("point_filter_num", p_pre->point_filter_num))
        p_pre->point_filter_num = 2;
    //是否提取特征点（FAST_LIO2默认不进行特征点提取）
    if(!node->get_parameter("feature_extract_enable", p_pre->feature_enabled))
        p_pre->feature_enabled = false;
    //是否输出调试log信息
    if(!node->get_parameter("runtime_pos_log_enable", runtime_pos_log))
        runtime_pos_log = 0;
    if(!node->get_parameter("mapping/extrinsic_est_en", extrinsic_est_en))
        extrinsic_est_en = true;
    // 是否将点云地图保存到PCD文件
    if(!node->get_parameter("pcd_save/pcd_save_en", pcd_save_en))
        pcd_save_en = false;
    // 每一个PCD文件保存多少个雷达帧（-1表示所有雷达帧都保存在一个PCD文件中）
    if(!node->get_parameter("pcd_save/interval", pcd_save_interval))
        pcd_save_interval = -1;
    //雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    if(!node->get_parameter("mapping/extrinsic_T", extrinT))
        extrinT = vector<double>(3, 0.0);
    //雷达相对于IMU的外参R
    if(!node->get_parameter("mapping/extrinsic_R", extrinR))
        extrinR = vector<double>(9, 0.0);
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;
    
    //初始化path的header（包括时间戳和帧id），path用于保存odemetry的路径
    path.header.stamp    = node->get_clock()->now();
    // path.header.frame_id = "camera_init";
    path.header.frame_id = "laser_link";

    /*** variables definition ***/
    //后面的代码中没有用到该变量
    int effect_feat_num = 0;
    //雷达总帧数
    int frame_num = 0;
    //后面的代码中没有用到该变量
    double deltaT, deltaR;
    //每帧平均的处理总时间
    double aver_time_consu = 0;
    //每帧中icp的平均时间
    double aver_time_icp = 0;
    //每帧中匹配的平均时间
    double aver_time_match = 0;
    //每帧中ikd-tree增量处理的平均时间
    double aver_time_incre = 0;
    //每帧中计算的平均时间
    double aver_time_solve = 0;
    //每帧中计算的平均时间（当H恒定时）
    double aver_time_const_H_time = 0;
    //后面的代码中没有用到该变量
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    //这里没用到
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    //将数组point_selected_surf内元素的值全部设为true，数组point_selected_surf用于选择平面点
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    //数组res_last内元素的值全部设置为-1000.0f，数组res_last用于平面拟合中
    memset(res_last, -1000.0f, sizeof(res_last));
    //VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为filter_size_surf_min
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    //VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为filter_size_map_min
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    //重复操作 没有必要
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    //从雷达到IMU的外参R和T
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

    //设置IMU的参数，对p_imu进行初始化，其中p_imu为ImuProcess的智能指针（ImuProcess是进行IMU处理的类）
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001); ////从epsi填充到epsi+22 也就是全部数组置0.001
    // 将函数地址传入kf对象中，用于接收特定于系统的模型及其差异
    // 作为一个维数变化的特征矩阵进行测量。
    // 通过一个函数（h_dyn_share_in）同时计算测量（z）、估计测量（h）、偏微分矩阵（h_x，h_v）和噪声协方差（R）。
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    //// 将调试log输出到文件中
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    // fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    // fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    // fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    // if (fout_pre && fout_out)
    //     cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    // else
    //     cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    //雷达点云的订阅器sub_pcl，订阅点云的topic
    if(p_pre->lidar_type == AVIA)
    {
        node->create_subscription<livox_ros_driver::msg::CustomMsg>(lid_topic, 20000, livox_pcl_cbk);
    }
    else
    {
        node->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, 20000, standard_pcl_cbk);
    }
    
    //IMU的订阅器sub_imu，订阅IMU的topic
    node->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 20000, imu_cbk);

    //发布当前正在扫描的点云，topic名字为/cloud_registered
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull = 
        node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 10000);

    //发布经过运动畸变校正到IMU坐标系的点云，topic名字为/cloud_registered_body
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body =
        node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 10000);
    
    //后面的代码中没有用到
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect =
        node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 10000);
    //后面的代码中没有用到
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap =
        node->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 10000);
    //发布当前里程计信息，topic名字为/Odometry
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped =
        node->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 10000);
    //发布里程计总的路径，topic名字为/path
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath =
        node->create_publisher<nav_msgs::msg::Path>("/path", 100000);

//------------------------------------------------------------------------------------------------------
    //中断处理函数，如果有中断信号（比如Ctrl+C），则执行第二个参数里面的SigHandle函数
    signal(SIGINT, SigHandle);
    //设置ROS程序主循环每次运行的时间至少为0.0002秒（5000Hz）
    rclcpp::Rate rate(5000);
    while (rclcpp::ok())
    {
        // 如果有中断产生，则结束主循环
        if (flg_exit) break;
        // ROS消息回调处理函数，放在ROS的主循环中
        rclcpp::spin_some(node);
        // 将激光雷达点云数据和IMU数据从缓存队列中取出，进行时间对齐，并保存到Measures中
        if(sync_packages(Measures)) 
        {
            // 激光雷达第一次扫描
            if (flg_first_scan)
            {
                // 记录激光雷达第一次扫描的时间
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            // 对IMU数据进行预处理，其中包含了点云畸变处理 前向传播 反向传播
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            //世界系下雷达坐标系的位置
            //下面式子的意义是W^p_L = W^p_I + W^R_I * I^t_L
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(node->get_logger(), "No point, skip this scan!\n");
                continue;//无点则跳过
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            //初始化ikdtree
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(node->get_logger(), "No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            //迭代卡尔曼滤波更新
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
