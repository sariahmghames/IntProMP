#include "GRASPberryArm.h"
#include "JMCController.h"
#include "SocketCANDevice.h"
//#include "CANalystIIDevice.h" test with CANalystII device
#include <dynamic_reconfigure/server.h>
#include <graspberry_robot/GraspberryArmConfig.h>  // generated after "making" in cfg folder

class ScaraArm : public GRASPberryArm {  // GRASPberryArm is a base class, ScaraArm is inherited from class GRASPberryArm, every object of type ScaraArm inherits the methods of GRASPberryArm
	enum State {  // list of enumerators
		UNINITIALISED,
		POSITION,
		HOMING
	};

	CANOpen::DevicePtr device; // CANOpen is a namespace and DevicePtr is a variable of type Ptr to class Device
    std::vector<CANOpen::JMCController*> joint_controller; //x, y, z axis controllers  ... JMCController is a class of the namespace CANOpen
    const std::vector<double> axis_ratio = {0.020, 20, 20}; //x, y, z axis ratios in 2pi/rev, 2pi/rev, meter/rev
    const std::vector<double> cob_id = {0x603, 0x602, 0x601}; //x, y, z axis ratios in metre/rev  ... cob_id ??
    const std::vector<uint8_t> homing_mode = {6, 6, 6};
    const double incprev = 4000.0; //encoder increments per revolution
    double time_out = 0.5;
    State state; // object of type enumerators list
    typedef dynamic_reconfigure::Server<graspberry_robot::GraspberryArmConfig> ReconfigureServer;
    boost::shared_ptr<ReconfigureServer> reconfigure_server;    

    /// parameters
    std::vector<double> profile_vel, profile_acc, profile_dec, homing_vel;
    bool homing_at_init;

    //
    std::vector<uint16_t> current_status = {0, 0, 0};
public:

	ScaraArm() : // constructor function that inherits from constructor function GRASPberryArm() and the private enum list state
		GRASPberryArm(), state(State::UNINITIALISED) {
		profile_vel = {0.0, 0.0, 0.0};
		profile_acc = {0.0, 0.0, 0.0};
		profile_dec = {0.0, 0.0, 0.0};
		homing_vel = {0.0, 0.0, 0.0};
		reconfigure_server = boost::make_shared<ReconfigureServer>();  // a constructor
		reconfigure_server->setCallback(boost::bind(&ScaraArm::reconfCallback, this, _1, _2));
	}

	~ScaraArm() { // destructor
		if (device) 
			device->Close();  // device is of type ptr to class Device of CANopen namespace
	}

	virtual void init() {

		device = std::make_shared<CANOpen::SocketCANDevice> (time_out); // SocketCANDevice is a member function of CANOpen ns and takes time_out as arg

		device->Init();  // CAN

		for (int i = 0; i < cob_id.size(); i++)
			joint_controller.push_back(new CANOpen::JMCController(device, cob_id[i]));

        //start remote node
		for (int i = 0; i < joint_controller.size(); i++)
			joint_controller[i]->StartRemoteNode();  // elements of vector joint_controller are pointers to class JMCController

		//enable controller
		for (int i = 0; i < joint_controller.size(); i++) {
            joint_controller[i]->ControlWord(CANOpen::JMCController::ControlWordCommand::Shutdown); //shutdown
            joint_controller[i]->ControlWord(CANOpen::JMCController::ControlWordCommand::SwitchOn); //switched on
            joint_controller[i]->ControlWord(CANOpen::JMCController::ControlWordCommand::EnableOperation); //Operation Enabled
        }

        //set position control mode
        for (int i = 0; i < joint_controller.size(); i++) {
        	joint_controller[i]->ModeOfOperation(CANOpen::JMCController::ModeOfOperationValue::PositionMode);
        	joint_controller[i]->ProfileVelocity(profile_vel[i]);  // from the CANopen header --> member function of class CiA402::Interface 
        	joint_controller[i]->ProfileAcceleration(profile_acc[i]);
        	joint_controller[i]->ProfileDeceleration(profile_dec[i]);
        }

        read();
    }

    virtual void zero() {
    	std::cerr << "Homing..." << std::endl;
    	state = State::HOMING;

    	for (int i = 0; i < joint_controller.size(); i++) {
    		joint_controller[i]->ModeOfOperation(CANOpen::JMCController::ModeOfOperationValue::HomingMode);
	      	joint_controller[i]->Write(CANOpen::CiA402::Command::ZeroReturnMode, homing_mode[i]);//6 ccw?  ... from CANopen::Interface base class of CiA402::Interface
	      	joint_controller[i]->Write(CANOpen::CiA402::Command::ZeroReturnSpeed, homing_vel[i]); //1.0 rps
	      	joint_controller[i]->Write(CANOpen::CiA402::Command::ZeroReturnAccDec, 20); //10 rpss
	      	joint_controller[i]->Write(CANOpen::CiA402::Command::ZeroReturnSwitchSpeed, 2); //0.5 rps	      	
	      	joint_controller[i]->ControlWord(CANOpen::JMCController::ControlWordCommand::NewLocation); //set new location			
	      }
	  }


	  virtual void read() {
	  	ros::Time now = ros::Time::now();


	  	//update the state
/*
	  	for (int i = 0; i < joint_controller.size(); i++) {
	  		pos[i] = joint_controller[i]->Position();
	  		vel[i] = joint_controller[i]->Velocity();
	  		eff[i] = 0.0;	  		
	  	} */


	  	for (int i = 0; i < joint_controller.size(); i++) {
	  		pos[i] = joint_controller[i]->Position()*axis_ratio[i]/incprev; //value in SI units [m]
	  		vel[i] = joint_controller[i]->Velocity()*axis_ratio[i]/10/incprev; //value in SI units [m/s]
	  		eff[i] = 0.0;	  		
	  	}


    	if (false)
	  	//show the status in binary format
	  	for (int i = 0; i < joint_controller.size(); i++) {
	  		uint16_t new_status = joint_controller[i]->StatusWord();
	  		if (new_status != current_status[i]) {
	  			current_status[i] = new_status;
	  			std::cerr << "status  " << i << ": " << std::bitset<16>(new_status) << std::endl;
	  		}
	  	}  

	  	period = now - last_update;
	  	last_update = now;
	  }

	  virtual void write() {

	  	//position controller
		for (int i = 0; i < joint_controller.size(); i++) {
			if (fabs(cmd[i] - pos[i]) > 1e-6) {  // error
				joint_controller[i]->SetPosition(cmd[i]);
			}	  		
		}
  	}

	  	void reconfCallback(graspberry_robot::GraspberryArmConfig& config, uint32_t) {

	  		//profile velocity
	  		std::vector<double> profile_vel_cfg = {config.profile_vel_0, config.profile_vel_1, config.profile_vel_2};

	  		for (int i = 0; i < profile_vel.size(); i++) {
	  			if (profile_vel[i] != profile_vel_cfg[i]) {
	  				profile_vel[i] = profile_vel_cfg[i];
	  				if (state != State::UNINITIALISED)
	  					joint_controller[i]->ProfileVelocity(profile_vel[i]);
	  			}
	  		}

	  		//profile accceleration
	  		std::vector<double> profile_acc_cfg = {config.profile_acc_0, config.profile_acc_1, config.profile_acc_2};

	  		for (int i = 0; i < profile_acc.size(); i++) {
	  			if (profile_acc[i] != profile_acc_cfg[i]) {
	  				profile_acc[i] = profile_acc_cfg[i];
	  				if (state != State::UNINITIALISED)
	  					joint_controller[i]->ProfileAcceleration(profile_acc[i]);
	  			}
	  		}

	  		//profile decelearion
	  		std::vector<double> profile_dec_cfg = {config.profile_dec_0, config.profile_dec_1, config.profile_dec_2};

	  		for (int i = 0; i < profile_dec.size(); i++) {
	  			if (profile_dec[i] != profile_dec_cfg[i]) {
	  				profile_dec[i] = profile_dec_cfg[i];
	  				if (state != State::UNINITIALISED)
	  					joint_controller[i]->ProfileDeceleration(profile_dec[i]);
	  			}
	  		}

	  		//homing at init
	  		homing_at_init = config.homing;

	  		//profile decelearion
	  		std::vector<double> homing_vel_cfg = {config.homing_vel_0, config.homing_vel_1, config.homing_vel_2};

	  		for (int i = 0; i < homing_vel.size(); i++) {
	  			if (homing_vel[i] != homing_vel_cfg[i]) {
	  				homing_vel[i] = homing_vel_cfg[i];
		      }
		  }	
		}
	};