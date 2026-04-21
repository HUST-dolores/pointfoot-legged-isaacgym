#include <webots/Robot.hpp>
#include <webots/Motor.hpp>
#include <webots/InertialUnit.hpp>
#include <webots/Keyboard.hpp>
#include <webots/TouchSensor.hpp>
#include <iostream>
#include <fstream>
#include <webots/PositionSensor.hpp>
#include <webots/Gyro.hpp>
#include <cmath>
#include <webots/Altimeter.hpp>
#include "MiniPID.h"
#include "MiniPID.cpp"
#include "WheeledBiped.hpp"
#include "joystick.h"
#include <lcm/lcm-cpp.hpp>
#include "lcm-types/cpp/imu_data.hpp"
#include "lcm-types/cpp/remote_data.hpp"
#include "lqr.h"
#include "pid_msg.hpp"
#include "state.hpp"
#include <time.h>
#include <webots/Supervisor.hpp>
#include <webots/Node.hpp>
#include <webots/Field.hpp>
#include <iomanip>

using namespace webots;
using namespace std;

#define TORQUE 100
#define MAX_TORQUE 5

class Two_legwheel_robot
{
private:    
    Supervisor* robot = new Supervisor();  
    int timeStep = (int)robot->getBasicTimeStep();
    Motor* l_hip_motor = robot->getMotor("l_hip_motor");
    Motor* r_hip_motor = robot->getMotor("r_hip_motor");
    Motor* l_knee_motor = robot->getMotor("l_knee_motor");
    Motor* r_knee_motor = robot->getMotor("r_knee_motor");
    Motor* l_wheel_motor = robot->getMotor("l_wheel_motor");
    Motor* r_wheel_motor = robot->getMotor("r_wheel_motor");
    InertialUnit* imu = robot->getInertialUnit("imu");
    Keyboard key_get;
    //xbox
    xbox_control xbox;
    xbox_control::xbox_map* _remote_data;
    //lcm
    lcm::LCM _imu_data;
    lcm::LCM _state_data;
    lcm::LCM _pid_data;
    imu_data imu_msgs;
    remote_data remote_msgs;
    pid_msg pid_msgs;
    state state_msgs;
    PositionSensor* l_hip_poi_sensor = robot->getPositionSensor("l_hip_poi_sensor");
    PositionSensor* r_hip_poi_sensor = robot->getPositionSensor("r_hip_poi_sensor");
    PositionSensor* l_knee_poi_sensor = robot->getPositionSensor("l_knee_poi_sensor");
    PositionSensor* r_knee_poi_sensor = robot->getPositionSensor("r_knee_poi_sensor");
    PositionSensor* l_wheel_poi_sensor = robot->getPositionSensor("l_wheel_poi_sensor");
    PositionSensor* r_wheel_poi_sensor = robot->getPositionSensor("r_wheel_poi_sensor");
    TouchSensor* r_touch_sensor = robot->getTouchSensor("r_touch_sensor");
    TouchSensor* l_touch_sensor = robot->getTouchSensor("l_touch_sensor");
    Altimeter* height_sensor_body = robot->getAltimeter("altimeter_body");
    Altimeter* height_sensor_left_wheel = robot->getAltimeter("altimeter_left_wheel");
    Altimeter* height_sensor_right_wheel = robot->getAltimeter("altimeter_right_wheel");
    //balance
    double v_avail,x_avail,turn_orientation = 0;
    float kp = 12, kd = 10, kp1 = 24.9, ki1 = 3, kd1 = 8.6;//1800 1700 23 3 6
    double differential_imux = 0, angle_imux = 0;
    double imux = 0;
    double v_sum_last = 0, v_sum = 0, v_current_last = 0, v_current = 0, v_differential = 0;
    double power = 0;
    double power_lwheel=0,power_rwheel=0,power_lknee=0,power_rknee=0,power_lhip=0,power_rhip=0;
    double v_avail_sum = 0;
    double angle_bias =0.20719;
    //负载辨识
    double zuoyou,qianhou,M_body=0,M_body_left=0,M_body_right=0;
    //height change
    double a_centri,delta_h,delta_rxh,delta_h_limit,delta_h_fina;
    int left_or_right;
    double yaw_target=0;
    // 目标倾角自动计算参数
    double balance_rod_length = 0.165632; // m
    double body_equiv_mass = 16.7;        // kg
    //jump

    double touch_floor = 1;
    bool count[20] = { 0 };
    double hip_poi_l = 0, knee_poi_l = 0;
    double hip_poi_r = 0, knee_poi_r = 0;
    bool jump_start = 0;
    double current_wheel_poi = 0;
    ofstream torque_test;
    WheeledBiped solve_func = WheeledBiped(0.3,0.3,0.1);
    double x_hip = 0;
    double y_hip = -0.1783;
    int if_add = 1;
    vector<double> T1;
    vector<double> T2;
        int i_T1;
    vector<vector<double>> T;
    int i_T = 0;
    int t=0;
    //lqr
    Eigen::Matrix<double,4,4> A;
    Eigen::Matrix<double,4,1> B;
    Eigen::Matrix<double,4,4> C;
    Eigen::Matrix<double,4,1> D;
    Eigen::Matrix<double,1,4> K;
    double dis_time = double(timeStep)/1000;
    Eigen::Matrix<double,4,1> x_state; 
    Eigen::Matrix<double,4,1> hope_state;  
    double pre_pitch = 0,pre_roll = 0,pre_yaw = 0;
    double pre_rhipangle = 1.309;       //是从蹬直为0,坐下那个方向为正
    double pre_lhipangle = 1.309; 
    double pre_rkneeangle = 2.618;      //是从180度为0,站最高为0，坐下那个方向为正
    double pre_lkneeangle = 2.618;
    // double pre_rhipangle = 0;
    // double pre_lhipangle = 0; 
    // double pre_rkneeangle = 0;
    // double pre_lkneeangle = 0;
    double pre_rwheelangle = 0;
    double pre_lwheelangle = 0;
    double pitch,roll,yaw, rhipangle, lhipangle, rkneeangle, lkneeangle, rwheelangle, lwheelangle;
    //pitch朝地面倒是正
    double pitchdot,rolldot, yawdot, rhipangledot, lhipangledot, rkneeangledot, 
           lkneeangledot, rwheelangledot, lwheelangledot;
    double circle=0;   //圈数
    double robotangle, robotangledot, robotvelocity;
    double velocity_sum, height;
    vector<Eigen::Matrix<double,4,1>> K_clump;
    double height_sum[10];
    clock_t first_time,second_time;
    double a1,a2,a3;
    int count_time = 0;
    double time_sum;//世界时间
    Node* target_node = nullptr;
    Field* target_translation = nullptr;
    Field* target_physics = nullptr;
    Field* target_mass = nullptr;

    // 负载轨迹参数
    bool load_traj_inited = false;
    bool load_traj_started = false;
    bool load_traj_enable = false;    // true: 负载按轨迹移动; false: 负载不自动移动
    double load_traj_start_time = 0.0;
    double load_center_x = 0.0, load_center_y = 0.0, load_center_z = 0.0;
    double load_last_x = 0.0, load_last_y = 0.0, load_last_z = 0.0;
    bool load_use_rectangle = true;   // true: 矩形轨迹; false: 圆轨迹
    double load_rect_length = 0.26;   // 矩形长(x方向)
    double load_rect_width  = 0.13;   // 矩形宽(y方向)
    double load_radius = 0.1;      // 圆半径(米)，可改
    double load_period = 8.0;      // 一圈8秒
    // 负载质量随时间切换
    bool load_mass_schedule_enable = true;     // true: 启用按时间改质量
    bool load_mass_switched = false;           // 防止重复切换
    double load_mass_switch_time = 7.0;        // 在7秒时切换
    double load_mass_before = 3;             // 切换前质量(kg) 注意不可以直接是0，因为质量不能为0
    double load_mass_after = 5;              // 切换后质量(kg)
    bool load_mass_call_reset_physics = false; // 如需重置物理状态可打开
    // 负载质量微调抖动：相邻步交替 +delta / -delta
    bool load_mass_dither_enable = false;
    double load_mass_dither_delta = 0.01;
    int load_mass_dither_every_n_steps = 30; // 每N步微调一次
    int load_mass_dither_step_counter = 0;
    int load_mass_dither_phase = 0;           // 0: +delta, 1: -delta 

    // 负载质量渐增：在固定时长内按步进频率将质量线性增加
    bool load_mass_ramp_enable = false;       // true: 启用“质量渐增”
    double load_mass_ramp_delta = 2.0;        // 总增量(kg)
    double load_mass_ramp_duration = 4.0;     // 持续时间(s)
    int load_mass_ramp_every_n_steps = 10;     // 每N步更新一次
    bool load_mass_ramp_trigger_by_time = true; // true: 按时间触发渐增
    double load_mass_ramp_trigger_time = 7.0;   // 触发时刻(s)
    bool load_mass_ramp_inited = false;
    bool load_mass_ramp_done = false;
    int load_mass_ramp_step_counter = 0;
    double load_mass_ramp_start_time = 0.0;
    double load_mass_ramp_start_mass = 0.0;
    ///为什么需要这功能？因为不知为何，如果不刷新质量，就会导致负载的位置的变化不更新到仿真引擎。原因应该和supervisor有关。
    //也就是说，如果不改变负载位置的话，以上部分可以设置为false。
    // 日志控制
    bool log_detail = true;          // true: 输出详细调试信息
    bool log_compact = true;          // true: 非详细模式下输出紧凑单行摘要
    int log_every_n_steps =180;       // 非详细模式下每N步输出一次
    int log_precision = 4;            // 紧凑日志小数位
    int log_step_counter = 0;
    bool log_now = false;

    // hope 倾角批量平均（每100步更新一次）
    int hope_tilt_avg_window = 30;
    int hope_tilt_avg_count = 0;
    double hope_tilt_avg_sum = 0.0;

    // hope 倾角惯性滤波（按 theta_target 变化速度自适应）
    bool hope_tilt_filter_inited = false;
    double hope_tilt_prev_theta_target = 0.0;
    double hope_tilt_filtered = 0.0;
    double hope_tilt_gain = -2.0;           // 目标倾角增益：hope_target = gain * theta_target
    double hope_tilt_rate_threshold = 0.8;  // rad/s，超过认为变化快
    double hope_tilt_alpha_normal = 0.15;   // 正常变化时滤波系数
    double hope_tilt_alpha_fast = 0.03;     // 快速变化时滤波系数（更“钝化”）

    // 方案F：位置误差 PID 额外力矩（基于 hope_state(2)-x_state(2)）
    bool use_position_pid_torque = false;  // true: 启用该方案并不调用 update_hope_tilt_from_load
    double pos_pid_kp = 25.0;
    double pos_pid_ki = 0.0;
    double pos_pid_kd = 1.0;
    double pos_pid_integral = 0.0;
    double pos_pid_integral_limit = 2.0;
    double pos_pid_prev_error = 0.0;
    bool pos_pid_inited = false;
    double power_pos_pid = 0.0;
public:
    Two_legwheel_robot();
    ~Two_legwheel_robot();
    int ifstep();
    bool if_floor();
    void ReceFromRemote();
    double forced_back(TouchSensor *touch_sensor);    
    void io_control(); 
    void balance();
    void height_change(double a1,double a2,double a3);
    void upforward_jumping();
    void control_jumping();
    void SendDriver();
    void data_to_lcm();
    void ReceFromLcm();
    void balancing();
    void update_hope_tilt_from_load();
    void get_K();
    void get_Q();
    void move_solid(double x, double y, double z);
};

Two_legwheel_robot::Two_legwheel_robot()
{
    imu->enable(timeStep);
    key_get.enable(timeStep);
    l_hip_poi_sensor->enable(timeStep);
    r_hip_poi_sensor->enable(timeStep);
    l_knee_poi_sensor->enable(timeStep);
    r_knee_poi_sensor->enable(timeStep);
    l_wheel_poi_sensor->enable(timeStep);
    r_wheel_poi_sensor->enable(timeStep);
    r_touch_sensor->enable(timeStep);
    l_touch_sensor->enable(timeStep);
    l_hip_motor->enableTorqueFeedback(timeStep);
    r_hip_motor->enableTorqueFeedback(timeStep);
    l_knee_motor->enableTorqueFeedback(timeStep);
    r_knee_motor->enableTorqueFeedback(timeStep);
    l_wheel_motor->enableTorqueFeedback(timeStep);
    r_wheel_motor->enableTorqueFeedback(timeStep);
    // l_wheel_motor->setPosition(INFINITY);
    // r_wheel_motor->setPosition(INFINITY);
    height_sensor_body->enable(timeStep);
    height_sensor_left_wheel->enable(timeStep);
    height_sensor_right_wheel->enable(timeStep);
    first_time = clock();
    second_time = clock();
    v_avail = 0;  
    x_avail = 0;
    height = 0.1872;  
    get_K();  //这个函数本身相当于参数设置，如果去除，会导致里面的参数是极大或极小值
    x_state << 0, 0, 0, 0;
    hope_state << 0, 0, 0, 0;
    delta_rxh = 0;
    time_sum=0;
    //T = solve_func.dynamic_func(13,-20,0.25,0.05);
    target_node = robot->getFromDef("load"); // world里 Solid 的 DEF 名
    if (target_node) {
        target_translation = target_node->getField("translation");
        target_physics = target_node->getField("physics");
    } 
    if (target_translation) {
        const double* p0 = target_translation->getSFVec3f();
        // 以仿真初始位置作为矩形轨迹起点（x为正的长边中点）
        load_center_x = p0[0];
        load_center_y = p0[1];
        load_center_z = p0[2];
           load_last_x = load_center_x;
           load_last_y = load_center_y;
           load_last_z = load_center_z;
        load_traj_started = false;
        load_traj_start_time = 0.0;
           cout << "[load_init] x=" << load_center_x << " y=" << load_center_y << " z=" << load_center_z << endl;
        load_traj_inited = true;
    }
    if (target_physics && target_physics->getSFNode()) {
        target_mass = target_physics->getSFNode()->getField("mass");
        if (target_mass) {
            target_mass->setSFFloat(load_mass_before);
            if (log_detail) {
                cout << "[load_mass] init_mass: " << load_mass_before
                     << " | switch_at: " << load_mass_switch_time
                     << "s -> " << load_mass_after << endl;
            }
        }
    }

}

Two_legwheel_robot::~Two_legwheel_robot(){
        delete robot;
    }

int Two_legwheel_robot::ifstep(){
    return robot->step(timeStep);
}

bool Two_legwheel_robot::if_floor(){
    if (forced_back(r_touch_sensor) + forced_back(l_touch_sensor) < 50)
    {
        touch_floor = 0;
        
    }
    else
        touch_floor = 1;
        

    return touch_floor;
}

void Two_legwheel_robot::ReceFromRemote()
{   
    _remote_data = xbox.print_word();
    if (_remote_data->lo == 1)
        throw "this is wrong!"; 
    v_avail -= (double)((_remote_data->ly < 200 && _remote_data->ly > -200)?0:_remote_data->ly) / 4000000;
    //x_avail = x_avail+v_avail*通讯时间;
    turn_orientation = (double)((_remote_data->lx < 200 && _remote_data->lx > -200)?0:_remote_data->lx) / 40000;
    // x_hip += (double)((_remote_data->rx < 200 && _remote_data->rx > -200)?0:_remote_data->rx) / 20000000;
    // y_hip += (double)((_remote_data->ry < 200 && _remote_data->ry > -200)?0:_remote_data->ry) / 20000000;
    // angle_bias += (double)_remote_data->a/100;
    // angle_bias -= (double)_remote_data->b/100;
    height -= (double)((_remote_data->ry < 200 && _remote_data->ry > -200)?0:_remote_data->ry) / 20000000;

    delta_rxh = (double)((_remote_data->rx < 200 && _remote_data->rx > -200)?0:_remote_data->rx) /300000;

    if (t == 0) jump_start = _remote_data->a;
    if(jump_start == 1) t = 200;
    if(t > 0) t--;
    // cout<<_remote_data->rx<<"\t"<<_remote_data->ry<<endl;
    //<<"\t"<<_remote_data->ly<<"\t"<<_remote_data->lx
    //32768八爪鱼 而且ly向下是正
}

double Two_legwheel_robot::forced_back(TouchSensor *touch_sensor)
{
    double force_3d[3];
    double force_back;
    for (int i = 0; i < 3; i++)
    {
        force_3d[i] = touch_sensor->getValues()[i];           
        force_back += force_3d[i] * force_3d[i];
    }
    force_back = sqrt(force_back);
    // cout << "FORCE_BACK= " << force_back  << endl;
    return force_back;    
} 

void Two_legwheel_robot::io_control()
{     
    int key;      
    key = key_get.getKey();        
    if (log_detail) cout << key << " ";               
    if (count_time == 0 && key != -1)
    {          
        // switch (key)
        // {
        //     case 315:       v_avail += 0.1;     break;//P
        //     case 317:       v_avail -= 0.1;     break;//DOWN
        //     case 89:        y_hip -= 0.01;      break;//y
        //     case 72:        y_hip += 0.01;      break;//h
        //     case 81:        kp += 0.01;         break;//q
        //     case 87:        kd += 0.01;         break;//w
        //     case 69:        kp1 += 0.01;        break;//e
        //     case 82:        ki1 += 0.01;        break;//r
        //     case 84:        kd1 += 0.01;        break;//t
        //     case 65:        kp -= 0.01;         break;//a
        //     case 83:        kd -= 0.01;         break;//s
        //     case 68:        kp1 -= 0.01;        break;//d
        //     case 70:        ki1 -= 0.01;        break;//f
        //     case 71:        kd1 -= 0.01;        break;//g
        //     case 90:        jump_start = 1;     break;//z
        // }
        switch (key)
        {
            case 315:       v_avail += 0.1;     break;//UP
            case 317:       v_avail -= 0.1;     break;//DOWN
            case 314:       yaw_target += 0.1;     break;//left
            case 316:       yaw_target -= 0.1;     break;//right

            case 89:        y_hip -= 0.01;      break;//y
            case 72:        y_hip += 0.01;      break;//h
            case 81:        height += 0.01;     break;//q
            case 87:        delta_rxh += 0.01;         break;//w
            case 69:        a3 += 0.01;         break;//e
            case 82:        ki1 += 0.01;        break;//r
            case 84:        kd1 += 0.01;        break;//t
            case 65:        height -= 0.01;     break;//a
            case 83:        delta_rxh -= 0.01;         break;//s
            case 68:        a3 -= 0.01;         break;//d
            case 70:        ki1 -= 0.01;        break;//f
            case 71:        kd1 -= 0.01;        break;//g
            case 90:        jump_start = 1;     break;//z
        }
        count_time = 20;
    }
    if (count_time > 0)     count_time--;
}

//pid
// void Two_legwheel_robot::balance(){
//     // if (key_get.getKey() == -1)
//     // {
//     //     if (fabs(v_avail) < 1)
//     //         v_avail = 0;
//     //     else
//     //         v_avail -= (v_avail / fabs(v_avail))/100;
//     // }
    
//     differential_imux = imu->getRollPitchYaw()[1] - imux;
//     v_sum = l_wheel_poi_sensor->getValue() / 2 + r_wheel_poi_sensor->getValue() / 2;
//     v_current = v_sum - v_sum_last;
//     v_differential = v_current - v_current_last;

//     /***********************************power***********************************************************/
//     // v_avail *= double(timeStep) / 1000;
//     // angle_imux = kp1 * (v_current - v_avail) + kd1 * v_differential + ki1 * (v_sum - v_avail_sum);
//     // angle_imux /= 100;
//     // power = (kp) * (imu->getRollPitchYaw()[1] - angle_bias + angle_imux) + kd * differential_imux;
//     // power = power / 100;
//     // if (power > MAX_TORQUE || power < -MAX_TORQUE)            power = power < 0 ? -MAX_TORQUE : MAX_TORQUE;
//     v_avail *= double(timeStep) / 1000;
//     //power = kp1 * (v_current - v_avail) + kd1 * v_differential + ki1 * (v_sum - v_avail_sum);
//     power += 31*(kp * (imu->getRollPitchYaw()[1] - angle_bias) + kd * differential_imux);//mgl=31
//     if (power > MAX_TORQUE || power < -MAX_TORQUE)            power = power < 0 ? -MAX_TORQUE : MAX_TORQUE;
//     /***********************************power***********************************************************/
//     //cout << imu->getRollPitchYaw()[1] << "\t" << power << endl;
//     //cout << angle_imux << "\t" << imu->getRollPitchYaw()[1] - angle_bias << endl;
//     // cout << v_current << "\t" << v_avail << endl;
//     // cout << angle_bias << endl;
//     l_wheel_motor->setTorque(power + turn_orientation);
//     r_wheel_motor->setTorque(power - turn_orientation);

//     imux = imu->getRollPitchYaw()[1];
//     v_current_last = v_current;
//     v_sum_last = v_sum;
//     v_avail_sum += v_avail;
//     v_avail /= double(timeStep) / 1000;
//}

//lqr
void Two_legwheel_robot::balancing()
{
    int i = 0;
    while(i < 9)
    {
        if(height >= height_sum[i] && height <= height_sum[i+1])
            break;
        i++;
        cout<<height_sum[i]<<endl;
    }
 
    if (i < 9)
    {   
        K = K_clump[i] + (height - height_sum[i])/(height_sum[i+1] - height_sum[i])*(K_clump[i+1] - K_clump[i]);
    }  
    else if(height <= height_sum[0]){
        
        height = height_sum[0];
        K = K_clump[0];
    }     
    else if(height >= height_sum[9]){
        height = height_sum[9];
        K = K_clump[9];
    }     
    
    // cout << K << endl;
    pitch = imu->getRollPitchYaw()[0];
    roll = imu->getRollPitchYaw()[1];
    yaw = imu->getRollPitchYaw()[2]-2*M_PI*circle;
    // cout<<"roll"<<"\t"<<roll<<endl;
    rhipangle = r_hip_poi_sensor->getValue();
    lhipangle = l_hip_poi_sensor->getValue();
    rkneeangle = r_knee_poi_sensor->getValue();
    lkneeangle = l_knee_poi_sensor->getValue();
    rwheelangle = r_wheel_poi_sensor->getValue();
    lwheelangle = l_wheel_poi_sensor->getValue();
    // cout << "rhipangle:" << rhipangle << endl;
    // cout << "rkneeangle:" << rkneeangle << endl;
    

    /*角速度单位为rad/s*/
    pitchdot = (pitch - pre_pitch)/timeStep*1000;
    rolldot = (roll - pre_roll)/timeStep*1000;
    yawdot = (yaw - pre_yaw)/timeStep*1000;
    rhipangledot = (rhipangle - pre_rhipangle)/timeStep*1000;
    lhipangledot = (lhipangle - pre_lhipangle)/timeStep*1000;
    rkneeangledot = (rkneeangle - pre_rkneeangle)/timeStep*1000;
    lkneeangledot = (lkneeangle - pre_lkneeangle)/timeStep*1000;
    rwheelangledot = (rwheelangle - pre_rwheelangle)/timeStep*1000;
    lwheelangledot = (lwheelangle - pre_lwheelangle)/timeStep*1000;
    // cout << "lwheel_dot:" << rwheelangledot << endl;

    x_state(2,0) = (rwheelangle+lwheelangle)/2*0.1;
    x_state(3,0) = (rwheelangledot+lwheelangledot)/2*0.1;

    /*计算得到现在机器人状态*/
    robotangle = (solve_func.get_angle(rhipangle,rkneeangle,roll) +
            solve_func.get_angle(lhipangle,lkneeangle,roll)/2);
    robotangledot = (solve_func.get_anglevelocity(rhipangle,rkneeangle,roll,rhipangledot,rkneeangledot,rolldot) +
            solve_func.get_anglevelocity(lhipangle,lkneeangle,roll,lhipangledot,lkneeangledot,rolldot)/2);
    robotvelocity = (solve_func.get_velocity(rwheelangledot) +
            solve_func.get_velocity(lwheelangledot))/2;
    velocity_sum += (robotvelocity - v_avail)*timeStep/1000;
    //height = solve_func.get_height(rhipangle);

    // if (log_detail) {
    //     cout << "[detail/angle] robotangle: " << robotangle
    //          << " | robotangledot: " << robotangledot << endl;
    // }

    x_state(0,0) = robotangle;
    x_state(1,0) = robotangledot;
    if (log_detail) {
        cout << "[detail/state] x_state: " << x_state.transpose()
             << " | hope_state: " << hope_state.transpose() << endl;
    }
    /*更新pre数据*/
    
    x_avail +=v_avail*timeStep/1000;
    hope_state(2,0) = x_avail;
    hope_state(3,0) = v_avail;
    if (!use_position_pid_torque) {
        update_hope_tilt_from_load();
    }
    power = -K*(x_state - hope_state);

    // 方案F：基于第三项误差的位置 PID 力矩，直接叠加到原力矩
    if (use_position_pid_torque) {
        const double dt = std::max(static_cast<double>(timeStep) / 1000.0, 1e-6);
        const double e = hope_state(2,0) - x_state(2,0);
        if (!pos_pid_inited) {
            pos_pid_prev_error = e;
            pos_pid_inited = true;
        }
        pos_pid_integral += e * dt;
        if (pos_pid_integral > pos_pid_integral_limit) pos_pid_integral = pos_pid_integral_limit;
        if (pos_pid_integral < -pos_pid_integral_limit) pos_pid_integral = -pos_pid_integral_limit;
        const double dedt = (e - pos_pid_prev_error) / dt;
        power_pos_pid = pos_pid_kp * e + pos_pid_ki * pos_pid_integral + pos_pid_kd * dedt;
        pos_pid_prev_error = e;

        power -= power_pos_pid;

        if (log_detail) {
            cout << "[detail/pos_pid] e=" << e
                 << " i=" << pos_pid_integral
                 << " de=" << dedt
                 << " pid_torque=" << power_pos_pid
                 << " | power_sum=" << power << endl;
        }
    } else {
        power_pos_pid = 0.0;
    }
    // if (log_detail) {
    //     cout << "[detail/lqr] K: " << K << endl;
    // }

    // cout << "x_state:" << x_state << "\thope_state:" << hope_state << endl;
    if (log_detail) cout << "[detail/control] power: " << power;

    MiniPID pidyaw=MiniPID(11,1,0.01);
    // cout << "yaw:" << pre_yaw <<  endl; 
    // cout << "yaw:" << yaw <<  endl; 
    if( yaw - pre_yaw > 3)
    {  
        circle++;   //右转超过半圈
        yaw=yaw-2*M_PI;
    }
    if(yaw- pre_yaw < -3)
    {  
        circle--;   //左转超过半圈
        yaw=yaw+2*M_PI;
    }

    double output_yaw=pidyaw.getOutput(yaw,yaw_target);

    if (log_detail) cout << " | circle: " << circle << endl; 
    // cout << "yaw_target:" << yaw_target <<  endl; 
    // cout << "yaw:" << pre_yaw <<  endl; 
    // cout << "yaw:" << yaw <<  endl; 
    if(power>10)
    {power=10;}
    l_wheel_motor->setTorque(power/2 - output_yaw);
    r_wheel_motor->setTorque(power/2 + output_yaw);

    power_lwheel=power/2 - output_yaw;
    power_rwheel=power/2 + output_yaw;
    

    // cout << "lkneeangle:" << lkneeangle <<  endl; 
    // cout << "lhipangle:" << lhipangle <<  endl; 
    // cout << "pitch:" << pitch <<  endl; 
    double  left_lieangle_thigh,right_lieangle_thigh;
    left_lieangle_thigh=solve_func.get_lieangle_thigh( M_PI-lkneeangle, lhipangle, pitch);
    right_lieangle_thigh=solve_func.get_lieangle_thigh( M_PI-rkneeangle, rhipangle, pitch);

    if (log_detail) {
        cout << "[detail/leg] left_lieangle_thigh: " << left_lieangle_thigh
             << " | right_lieangle_thigh: " << right_lieangle_thigh << endl;
    }

    power_lknee=l_knee_motor->getTorqueFeedback();
    power_rknee=r_knee_motor->getTorqueFeedback();
    power_lhip=l_hip_motor->getTorqueFeedback();
    power_rhip=r_hip_motor->getTorqueFeedback(); 
    M_body_left=-(power_lknee+power_lhip)/0.3/9.8/cos(left_lieangle_thigh)-8-0.008;
    M_body_right=-(power_rknee+power_rhip)/0.3/9.8/cos(right_lieangle_thigh)-8+-0.008;
    M_body=(M_body_left+M_body_right);
    if (M_body < 0.0) M_body = 0.0;
    const double M_body_safe = (M_body > 1e-6) ? M_body : 1e-6;
    zuoyou=M_body_left/M_body_safe*0.4-0.2;
    qianhou=((power_rhip+power_lhip)/M_body_safe/9.8/sin(pitch)-0.65)*tan(pitch);

    if (log_detail) {
           cout << "[detail/load_id] x_coordinate: " << zuoyou
               << " | y_coordinate: " << qianhou
             << " | M_body: " << M_body << endl;
        cout << "power_lhip: " << power_lhip
             << " | left_M_body: " << M_body_left
             << " | right_M_body: " << M_body_right << endl;
    }

    //0.3是大腿腿长
    // cout << "turn_orientation:" << turn_orientation << endl;
    pre_pitch = pitch;    //更新数据，这部分应该在最后
    pre_roll = roll;
    pre_yaw = yaw;
    pre_rhipangle = rhipangle;
    pre_lhipangle = lhipangle;
    pre_rkneeangle = rkneeangle;
    pre_lkneeangle = lkneeangle;
    pre_rwheelangle = rwheelangle;
    pre_lwheelangle = lwheelangle;
}

// void Two_legwheel_robot::height_change(double a1,double a2,double a3)
// {  
//     //if (_remote_data->rb == 1)    height = 0.3172;//0.1872 
//     double theta_need_l[2],theta_need_r[2]; //数值无所谓，为下个函数提供对象


//     a_centri = (lwheelangledot*0.1+rwheelangledot*0.1)*
//                       (lwheelangledot*0.1-rwheelangledot*0.1)/
//                       (4*0.2);       //0.2是两轮间距的一半
//     left_or_right = (lwheelangledot+1>rwheelangledot)?1:-1;  
//     //判断左转还是右转，因为delta_h是平方，没有了正负.+1防止0附近跳跃
//     //delta_h = height*a_centri*a_centri/(a_centri*a_centri+96.2361)+delta_rxh;
//     //这里做错了，应该和高度没关系
//     delta_h = 0.4/9.81*a_centri+delta_rxh;
//     //这是正确的，好像这样也不用判断左转右转了。
//     //cout << "delta_rxh:" << delta_rxh<< "\tleft_or_right:" << left_or_right<<endl;
//     delta_h_limit = (0.53-height > height-0.1873)?height-0.1873:0.53-height ;
//     delta_h_fina = (fabs(delta_h)>delta_h_limit)?(delta_h>0?delta_h_limit:-delta_h_limit):delta_h;
//     //里面那个判断负责判断高度变化限制的正负
//      cout << "delta_h:" << delta_h << "\tdelta_h_limit:" << delta_h_limit << 
//               "\tdelta_h_fina:" << delta_h_fina << endl;  
//     //去掉fina会导致程序更简洁但是不好调试
    
//     imu_msgs.roll = imu->getRollPitchYaw()[0];   
//     imu_msgs.pitch = imu->getRollPitchYaw()[1];  
//     imu_msgs.yaw = imu->getRollPitchYaw()[2];  


//     solve_func.get_needtheta(height+0.5*delta_h_fina, theta_need_l);     
//     cout << "height:" << height << "\trb:" << _remote_data->rb << endl;    
//     hip_poi_l = theta_need_l[0];
//     knee_poi_l = theta_need_l[1];  
//     solve_func.get_needtheta(height-0.5*delta_h_fina, theta_need_r); 
//     hip_poi_r = theta_need_r[0];
//     knee_poi_r = theta_need_r[1]; 

//     // cout << "hip:" << hip_poi << "\tknee:" << knee_poi << endl;  
//     l_hip_motor->setPosition(hip_poi_l);
//     l_knee_motor->setPosition(knee_poi_l);
//     r_hip_motor->setPosition(hip_poi_r);
//     r_knee_motor->setPosition(knee_poi_r);
//     // l_knee_motor->setControlPID(a1,a2,a3);
//     // r_knee_motor->setControlPID(a1,a2,a3);
//     // l_hip_motor->setControlPID(a1,a2,a3);
//     // r_hip_motor->setControlPID(a1,a2,a3);
// }

//有离心力平衡的高度改变

void Two_legwheel_robot::height_change(double a1,double a2,double a3)
{  
    //if (_remote_data->rb == 1)    height = 0.3172;//0.1872 
    double theta_need_l[2],theta_need_r[2]; //数值无所谓，为下个函数提供对象


    a_centri = (lwheelangledot*0.1+rwheelangledot*0.1)*
                      (lwheelangledot*0.1-rwheelangledot*0.1)/
                      (4*0.2);       //0.2是两轮间距的一半
    left_or_right = (lwheelangledot+1>rwheelangledot)?1:-1;  
    //判断左转还是右转，因为delta_h是平方，没有了正负.+1防止0附近跳跃
    //delta_h = height*a_centri*a_centri/(a_centri*a_centri+96.2361)+delta_rxh;
    //这里做错了，应该和高度没关系
    delta_h = 0.4/9.81*a_centri+delta_rxh;
    //这是正确的，好像这样也不用判断左转右转了。
    //cout << "delta_rxh:" << delta_rxh<< "\tleft_or_right:" << left_or_right<<endl;
    delta_h_limit = (0.53-height > height-0.1873)?height-0.1873:0.53-height ;
    delta_h_fina = (fabs(delta_h)>delta_h_limit)?(delta_h>0?delta_h_limit:-delta_h_limit):delta_h;
    //里面那个判断负责判断高度变化限制的正负
     //cout << "delta_h:" << delta_h << "\tdelta_h_limit:" << delta_h_limit << 
     //         "\tdelta_h_fina:" << delta_h_fina << endl;  
    //去掉fina会导致程序更简洁但是不好调试
    double roll_target=delta_h_fina/2/0.2;//0.2是两轮间距的一半

    MiniPID pidroll=MiniPID(2,1,0);
    //set any other PID configuration options here. 
    double output_roll=pidroll.getOutput(imu->getRollPitchYaw()[0],roll_target);


    if (log_detail) {
        cout << "[detail/height] output_roll: " << output_roll
             << " | height: " << height
             << " | rb: " << _remote_data->rb << endl;
    }

    solve_func.get_needtheta(height+0.5*output_roll, theta_need_l);   

    // 已在上方 [detail/height] 合并输出
    hip_poi_l = theta_need_l[0];
    knee_poi_l = theta_need_l[1];  
    solve_func.get_needtheta(height-0.5*output_roll, theta_need_r); 
    hip_poi_r = theta_need_r[0];
    knee_poi_r = theta_need_r[1]; 

    // cout << "hip:" << hip_poi << "\tknee:" << knee_poi << endl;  
    l_hip_motor->setPosition(hip_poi_l);
    l_knee_motor->setPosition(knee_poi_l);
    r_hip_motor->setPosition(hip_poi_r);
    r_knee_motor->setPosition(knee_poi_r);
    // l_knee_motor->setControlPID(a1,a2,a3);
    // r_knee_motor->setControlPID(a1,a2,a3);
    // l_hip_motor->setControlPID(a1,a2,a3);
    // r_hip_motor->setControlPID(a1,a2,a3);
}


void Two_legwheel_robot::upforward_jumping()
    {      
        if (jump_start == 1)
        {
            height = 0.1783;
            height_change(5,0.3,2);           
            count[0] = 1;  
            first_time = clock();   
            jump_start = 0;       
        }

        if (count[0] == 1 && l_knee_poi_sensor->getValue() > 2.49)
        {            
            count[0] = 0;
            count[1] = 1;
            current_wheel_poi = l_wheel_poi_sensor->getValue() / 2 + r_wheel_poi_sensor->getValue() / 2;
        }
        if (count[1] == 1 )
        {    
            l_knee_motor->setTorque(-25);
            r_knee_motor->setTorque(-25);
            l_hip_motor->setPosition(l_knee_poi_sensor->getValue()/2);
            r_hip_motor->setPosition(l_knee_poi_sensor->getValue()/2);
            l_wheel_motor->setPosition(current_wheel_poi - 1);
            l_wheel_motor->setVelocity(10);
            r_wheel_motor->setPosition(current_wheel_poi - 1);
            r_wheel_motor->setVelocity(10);            
            count[2] = 1;            
        }
        if (count[2] == 1 && l_knee_poi_sensor->getValue() < 1.5)
        {       
            height = 0.22;     
            height_change(10,10,10);
            count[1] = 0;            
            count[3] = 1;     
        }
        if (count[3] == 1 && forced_back(l_touch_sensor) + forced_back(r_touch_sensor) < 10)
        {
            count[2] = 0;
            count[4] = 1;  
        }
        if (count[4] == 1 && forced_back(l_touch_sensor) + forced_back(r_touch_sensor) > 90)
        {            
            height = 0.22;
            height_change(10,10,10);
            l_knee_motor->setAvailableTorque(60);
            r_knee_motor->setAvailableTorque(60);
            count[3] = 0;
            count[4] = 0; 
            second_time = clock(); 
            pre_roll = imu->getRollPitchYaw()[1];
            pre_rhipangle = r_hip_poi_sensor->getValue();
            pre_lhipangle = l_hip_poi_sensor->getValue();
            pre_rkneeangle = r_knee_poi_sensor->getValue();
            pre_lkneeangle = l_knee_poi_sensor->getValue();
            pre_rwheelangle = r_wheel_poi_sensor->getValue();
            pre_lwheelangle = l_wheel_poi_sensor->getValue();          
        }        
        for (int i = 0; i < 20; i++)
        {
            if (log_detail) cout << count[i] << " ";
        }
        if (log_detail) cout << endl << endl;
    }

void Two_legwheel_robot::control_jumping()
{
    if (i_T < T.size())
    {
        l_wheel_motor->setTorque(T[i_T][0]);
        r_wheel_motor->setTorque(T[i_T][0]);
        l_knee_motor->setTorque(T[i_T][1]);
        r_knee_motor->setTorque(T[i_T][1]);
        l_hip_motor->setTorque(T[i_T][2]);
        r_hip_motor->setTorque(T[i_T][2]);
    }
}

void Two_legwheel_robot::SendDriver()
{
    log_step_counter++;
    if (log_detail)
        log_now = true;
    else
        log_now = (log_every_n_steps > 0) ? ((log_step_counter % log_every_n_steps) == 0) : false;

    io_control();
    if (log_detail) {
        cout << "[detail/base] hip: " << l_hip_poi_sensor->getValue()
             << " | knee: " << l_knee_poi_sensor->getValue();
    }
    //cout << "\tsecond:" << second_time << "\tfirst:" << first_time << "\tdiff:" << (float)(second_time - first_time)/CLOCKS_PER_SEC;
    if (log_detail) cout << " | lwheel: " << l_wheel_poi_sensor->getValue() << " | rwheel: " << r_wheel_poi_sensor->getValue();
    if (log_detail) cout << " | current_wheel: " << current_wheel_poi << endl;
    time_sum += static_cast<double>(timeStep) / 1000.0;
    if (log_detail) cout << "[detail/time] time_sum: " << time_sum << endl;

    // 按时间自动改变负载质量（默认1s: 1kg -> 3kg）
    if (load_mass_schedule_enable && !load_mass_switched && target_mass && time_sum >= load_mass_switch_time) {
        target_mass->setSFFloat(load_mass_after);
        load_mass_switched = true;
        if (load_mass_call_reset_physics && target_node) {
            target_node->resetPhysics();
        }
        cout << "[load_mass] switched_at: " << time_sum
             << "s | mass: " << load_mass_after
             << (load_mass_call_reset_physics ? " | resetPhysics: on" : " | resetPhysics: off")
             << endl;
    }

    // 质量渐增：在 load_mass_ramp_duration 秒内累计增加 load_mass_ramp_delta，且每 load_mass_ramp_every_n_steps 步更新一次
    if (load_mass_ramp_enable && target_mass) {
        const bool ramp_triggered = (!load_mass_ramp_trigger_by_time) || (time_sum >= load_mass_ramp_trigger_time);

        if (!load_mass_ramp_inited && ramp_triggered) {
            load_mass_ramp_inited = true;
            load_mass_ramp_done = false;
            load_mass_ramp_step_counter = 0;
            load_mass_ramp_start_time = time_sum;
            load_mass_ramp_start_mass = target_mass->getSFFloat();
            if (log_detail) {
                cout << "[load_mass_ramp] start_mass: " << load_mass_ramp_start_mass
                     << " | trigger_t: " << load_mass_ramp_trigger_time
                     << " | delta: +" << load_mass_ramp_delta
                     << "kg | duration: " << load_mass_ramp_duration
                     << "s | every_n_steps: " << load_mass_ramp_every_n_steps << endl;
            }
        }

        if (load_mass_ramp_inited && !load_mass_ramp_done) {
            load_mass_ramp_step_counter++;
            const double duration_safe = (load_mass_ramp_duration > 1e-9) ? load_mass_ramp_duration : 1e-9;
            const double elapsed = time_sum - load_mass_ramp_start_time;
            const double ratio = std::max(0.0, std::min(1.0, elapsed / duration_safe));
            const int step_n = (load_mass_ramp_every_n_steps > 0) ? load_mass_ramp_every_n_steps : 1;

            // 每N步更新一次；达到终点时强制写最终值
            if ((load_mass_ramp_step_counter % step_n) == 0 || ratio >= 1.0) {
                const double m_next = load_mass_ramp_start_mass + load_mass_ramp_delta * ratio;
                target_mass->setSFFloat(m_next);
                if (log_detail) {
                    cout << "[load_mass_ramp] elapsed: " << elapsed
                         << "s | ratio: " << ratio
                         << " | m: " << m_next << endl;
                }
            }

            if (ratio >= 1.0) {
                load_mass_ramp_done = true;
                if (log_detail) {
                    cout << "[load_mass_ramp] done | final_mass: "
                         << (load_mass_ramp_start_mass + load_mass_ramp_delta) << endl;
                }
            }
        }
    }

    // 每N个步长进行一次质量微调：相邻次交替 +delta / -delta
    if (load_mass_dither_enable && target_mass) {
        load_mass_dither_step_counter++;
        const int dither_n = (load_mass_dither_every_n_steps > 0) ? load_mass_dither_every_n_steps : 1;
        if ((load_mass_dither_step_counter % dither_n) == 0) {
            const double m_now = target_mass->getSFFloat();
            const double sign = (load_mass_dither_phase == 0) ? 1.0 : -1.0;
            const double m_next = m_now + sign * load_mass_dither_delta;
            target_mass->setSFFloat(m_next);
            load_mass_dither_phase = 1 - load_mass_dither_phase;
            if (log_detail) {
                cout << "[load_mass_dither] m: " << m_now
                     << " -> " << m_next
                     << " | sign: " << (sign > 0 ? "+" : "-")
                     << " | every_n_steps: " << dither_n << endl;
            }
        }
    }
    // 每步更新负载位置
    if (load_traj_inited && load_traj_enable) {
        if (!load_traj_started) {
            load_traj_started = true;
            load_traj_start_time = time_sum;
        }
        double traj_t = time_sum - load_traj_start_time;

        double x = load_center_x;
        double y = load_center_y;
        double z = load_center_z;  // z固定

        if (load_period > 0.0) {
            if (load_use_rectangle) {
                // 在xy平面做矩形运动：起点为“x为正的长边中点”，按周长匀速绕行
                double perimeter = 2.0 * (load_rect_length + load_rect_width);
                double speed = perimeter / load_period;
                // 将起点相位平移到右侧长边中点（相当于从右下角走了L/2）
                double s = fmod(speed * traj_t + 0.5 * load_rect_length, perimeter);

                // 约定：长边平行y轴；load_center_* 是右侧长边中点
                const double right_x = load_center_x;
                const double y_mid = load_center_y;
                const double y_bottom = y_mid - 0.5 * load_rect_length;
                const double y_top = y_mid + 0.5 * load_rect_length;
                const double left_x = right_x - load_rect_width;

                if (s < load_rect_length) {
                    x = right_x;
                    y = y_bottom + s;
                } else if (s < load_rect_length + load_rect_width) {
                    x = right_x - (s - load_rect_length);
                    y = y_top;
                } else if (s < 2.0 * load_rect_length + load_rect_width) {
                    x = left_x;
                    y = y_top - (s - (load_rect_length + load_rect_width));
                } else {
                    x = left_x + (s - (2.0 * load_rect_length + load_rect_width));
                    y = y_bottom;
                }

                 if (log_detail) {
                    cout << "[load_traj_rect] t=" << time_sum
                        << " traj_t=" << traj_t
                        << " period=" << load_period
                        << " x=" << x
                        << " y=" << y
                        << " z=" << z << endl;
                 }
            } else {
                // 保留圆轨迹模式
                double omega = 2.0 * M_PI / load_period;
                 double theta = omega * traj_t;
                x = load_center_x + load_radius * cos(theta);
                y = load_center_y + load_radius * sin(theta);

                if (log_detail) {
                    cout << "[load_traj_circle] t=" << time_sum
                         << " traj_t=" << traj_t
                         << " omega=" << omega
                         << " theta=" << theta
                         << " x=" << x
                         << " y=" << y
                         << " z=" << z << endl;
                }
            }
        } else {
            if (log_detail) cout << "[load_traj] invalid load_period=" << load_period << endl;
        }

        load_last_x = x;
        load_last_y = y;
        load_last_z = z;
        move_solid(x, y, z);
    } else if (load_traj_inited && !load_traj_enable) {
        // 禁用轨迹时，不移动负载；并重置起始状态，保证再次启用时从当前位置平滑开始
        load_traj_started = false;
        const double* p_now = target_translation->getSFVec3f();
        load_center_x = p_now[0];
        load_center_y = p_now[1];
        load_center_z = p_now[2];
        load_last_x = load_center_x;
        load_last_y = load_center_y;
        load_last_z = load_center_z;
    }

    //if (if_floor() == 1 )//&& ((float)(second_time - first_time)/CLOCKS_PER_SEC > 0.015 || fabs((float)(second_time - first_time)/CLOCKS_PER_SEC) < 0.001 )
    if ( if_floor() == 1 )
    {

        height_change(0,0,0);

        //height_change(13,0.3,2);
        //printf("a1:%f\ta2:%f\ta3:%f\n",a1,a2,a3);
        balancing();
    }

    if (log_now && log_compact) {
        cout << fixed << setprecision(log_precision)
             << "[state] t=" << time_sum
             << " | hip=" << l_hip_poi_sensor->getValue()
             << " knee=" << l_knee_poi_sensor->getValue()
             << " | lw=" << l_wheel_poi_sensor->getValue()
             << " rw=" << r_wheel_poi_sensor->getValue()
             << " | P=" << power
             << " | floor=" << touch_floor
             << " | load(" << (load_traj_enable ? (load_use_rectangle ? "rect" : "circle") : "off")
             << ")=" << load_last_x << "," << load_last_y << "," << load_last_z
             << endl;
    }
    //cout << wl_robot.y_hip << endl;
    // cout << "l_wheel_poi:" << lwheelangle << "\tr_wheel_poi:" << rwheelangle << endl;
    //cout << "l_force:" << forced_back(l_touch_sensor) << "\tjump_start:" << jump_start << endl;
    //upforward_jumping();
    // wl_robot.control_jumping();
}

void Two_legwheel_robot::data_to_lcm()
{
    imu_msgs.roll = imu->getRollPitchYaw()[0];   
    imu_msgs.pitch = imu->getRollPitchYaw()[1];  
    imu_msgs.yaw = imu->getRollPitchYaw()[2];  
    
    pid_msgs.kp = height_sensor_body->getValue();
    pid_msgs.kd = kd;
    pid_msgs.kp1 = kp1;
    pid_msgs.ki1 = ki1;
    pid_msgs.kd1 = kd1;
    
    state_msgs.x_rwheel = r_wheel_poi_sensor->getValue()*0.1;
    state_msgs.x_lwheel = l_wheel_poi_sensor->getValue()*0.1;//R
    state_msgs.x_rhip = r_hip_poi_sensor->getValue();
    state_msgs.x_lhip = l_hip_poi_sensor->getValue();
    state_msgs.x_rknee = r_knee_poi_sensor->getValue();
    state_msgs.x_lknee = l_knee_poi_sensor->getValue();


    state_msgs.v_rwheel = rwheelangledot*0.1;
    state_msgs.v_lwheel = lwheelangledot*0.1;//R
    state_msgs.v_rhip  = rhipangledot;
    state_msgs.v_lhip  = lhipangledot;
    state_msgs.v_rknee = rkneeangledot;
    state_msgs.v_lknee = lkneeangledot; 

    state_msgs.angle = robotangle;
    state_msgs.angledot = robotangledot;

    state_msgs.x_hope  = hope_state(2,0) ; 
    state_msgs.v_hope  = hope_state(3,0) ; 
    state_msgs.angle_hope = hope_state(0,0) ; 
    state_msgs.angledot_hope = hope_state(1,0) ; 
    state_msgs.time_sum  = time_sum;

    state_msgs.power = power;
    state_msgs.power_lwheel = power_lwheel;
    state_msgs.power_rwheel = power_rwheel;
    state_msgs.power_lknee = power_lknee;
    state_msgs.power_rknee = power_rknee;
    state_msgs.power_lhip = power_lhip;
    state_msgs.power_rhip = power_rhip;
    state_msgs.if_floor = touch_floor;

    state_msgs.v_hope_left = lwheelangledot*0.1;
    state_msgs.v_hope_right = rwheelangledot*0.1;
    state_msgs.a_centri = a_centri;        //减少全局变量
    state_msgs.left_or_right = left_or_right;
    state_msgs.delta_rxh = delta_rxh;
    state_msgs.delta_h=delta_h;
    state_msgs.delta_h_limit=delta_h_limit;
    state_msgs.delta_h_fina=delta_h_fina;
    state_msgs.height=height; //很明显，左右腿高度分别等于height+-0.5*delta_h_fina

    state_msgs.zuoyou = zuoyou;
    state_msgs.qianhou = qianhou;
    state_msgs.M_body = M_body;        //减少全局变量
    state_msgs.M_body_left  = M_body_left;
    state_msgs.M_body_right = M_body_right;
    state_msgs.pendulum_angle = x_state(0,0);
}

void Two_legwheel_robot::ReceFromLcm()
{
    data_to_lcm();
    _pid_data.publish("pid_data",&pid_msgs);
    _imu_data.publish("imu_data",&imu_msgs);
    _state_data.publish("state_msgs",&state_msgs);
}

void Two_legwheel_robot::get_K()
{
   double l_start = 0.1872;
   double l_end = 0.53;
   for (int i = 0;i < 10;i++)
   {
    height_sum[i] = l_start + (double)i/10 * (l_end - l_start);
    solve_func.get_state_function(A,B,C,D,height_sum[i]);
    // cout << "height_sum:" << height_sum[i] << "\nA:\n" << A << "\nB:\n" << B << endl;
    K_clump.push_back(LQR_get_k(A,B,dis_time));
        cout << "height_sum:" << height_sum[i]
            << "\tdis_time:" << dis_time
            << "\tK:" << K_clump[i] << endl;
   }
}
void Two_legwheel_robot::get_Q()
{






}

void Two_legwheel_robot::move_solid(double x, double y, double z)
{
    if (!target_translation) return;
    const double p[3] = {x, y, z};
    target_translation->setSFVec3f(p);
}

void Two_legwheel_robot::update_hope_tilt_from_load()
{   
    if (balance_rod_length <= 1e-9)
        return;

    // 方案E（当前启用，对照组/完美组）：使用真值质量 target_mass

    // const double load_mass_truth = target_mass ? target_mass->getSFFloat() : load_mass_before;
    // const double total_mass = load_mass_truth + body_equiv_mass;
    // if (total_mass <= 1e-9)
    //     return;

    // // // 在本函数内基于真值质量重新计算临时 qianhou_（改全局 qianhou）
    // const double load_mass_truth_safe = (load_mass_truth > 1e-6) ? load_mass_truth : 1e-6;
    // const double sin_pitch_safe = (fabs(sin(pitch)) > 1e-6) ? sin(pitch) : (sin(pitch) >= 0.0 ? 1e-6 : -1e-6);
    // const double qianhou_ = ((power_rhip + power_lhip) / load_mass_truth_safe / 9.8 / sin_pitch_safe - 0.65) * tan(pitch);
    // qianhou = qianhou_;
    // // M_body= load_mass_truth_safe;
    // const double com = load_last_x * load_mass_truth / total_mass;
    // const double theta_target = atan(com / balance_rod_length);
    // hope_state(0,0) = -2 * theta_target;

    // ===== 历史方案保留（用于对比） =====
    // 方案A：M_body 估计质量（旧输入源）
    const double total_mass = M_body + body_equiv_mass;
    if (total_mass <= 1e-9 || balance_rod_length <= 1e-9)
        return;

    const double com = qianhou * M_body / total_mass;
    const double theta_target = atan(com / balance_rod_length);
    hope_state(0,0) = -2 * theta_target;



    // 方案B：使用 x_state 第三项（索引 2）
    // hope_state(0,0) = -x_state(2,0) / 4.0;

    // 方案C：固定窗口平均（旧方案，便于回滚对比）
    // const double total_mass = M_body + body_equiv_mass;
    // if (total_mass <= 1e-9 || balance_rod_length <= 1e-9)
    //     return;


    // const double com = qianhou * M_body / total_mass;
    // const double theta_target = atan(com / balance_rod_length);

    // const double hope_tilt_raw = -2 * theta_target;
    // hope_tilt_avg_sum += hope_tilt_raw;
    // hope_tilt_avg_count++;
    // if (hope_tilt_avg_count >= hope_tilt_avg_window) {
    //     hope_state(0,0) = hope_tilt_avg_sum / static_cast<double>(hope_tilt_avg_count);
    //     hope_tilt_avg_sum = 0.0;
    //     hope_tilt_avg_count = 0;
    // }

    // ==================================

    // 方案D：按 theta_target 变化速度的自适应惯性滤波（历史方案，保留注释）
    // const double total_mass = M_body + body_equiv_mass;
    // if (total_mass <= 1e-9 || balance_rod_length <= 1e-9)
    //     return;


    // const double com = qianhou * M_body / total_mass;
    // const double theta_target = atan(com / balance_rod_length);



    // const double dt = std::max(static_cast<double>(timeStep) / 1000.0, 1e-6);
    // const double hope_tilt_target = hope_tilt_gain * theta_target;
    
    // if (!hope_tilt_filter_inited) {
    //     hope_tilt_filtered = hope_tilt_target;
    //     hope_tilt_prev_theta_target = theta_target;
    //     hope_tilt_filter_inited = true;
    // }
    
    // const double theta_rate = fabs(theta_target - hope_tilt_prev_theta_target) / dt;
    // const double alpha = (theta_rate > hope_tilt_rate_threshold) ? hope_tilt_alpha_fast : hope_tilt_alpha_normal;
    
    // hope_tilt_filtered += alpha * (hope_tilt_target - hope_tilt_filtered);
    // hope_state(0,0) = hope_tilt_filtered;
    // hope_tilt_prev_theta_target = theta_target;

    


    // 方案E（当前启用）：不滤波，直接赋值（仅质量来源改为 target_mass）
    // 与方案A唯一区别：
    // - 方案A 使用 M_body 参与 com/theta_target 计算
    // - 方案E 使用 target_mass(真值) 参与 com/theta_target 计算


    if (log_detail) {
        cout << "[detail/hope] load_last_x: " << load_last_x
             << " | com: " << com
             << " | theta_target: " << theta_target<< endl;
            //  << " | theta_target: " << theta_target;
            //  << " | hope_tilt: " << hope_state(0,0) << endl;
    }
}