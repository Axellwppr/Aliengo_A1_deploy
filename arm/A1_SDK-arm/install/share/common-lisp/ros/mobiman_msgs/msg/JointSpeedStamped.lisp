; Auto-generated. Do not edit!


(cl:in-package mobiman_msgs-msg)


;//! \htmlinclude JointSpeedStamped.msg.html

(cl:defclass <JointSpeedStamped> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (speed
    :reader speed
    :initarg :speed
    :type mobiman_msgs-msg:JointSpeed
    :initform (cl:make-instance 'mobiman_msgs-msg:JointSpeed)))
)

(cl:defclass JointSpeedStamped (<JointSpeedStamped>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <JointSpeedStamped>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'JointSpeedStamped)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name mobiman_msgs-msg:<JointSpeedStamped> is deprecated: use mobiman_msgs-msg:JointSpeedStamped instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <JointSpeedStamped>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mobiman_msgs-msg:header-val is deprecated.  Use mobiman_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'speed-val :lambda-list '(m))
(cl:defmethod speed-val ((m <JointSpeedStamped>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mobiman_msgs-msg:speed-val is deprecated.  Use mobiman_msgs-msg:speed instead.")
  (speed m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <JointSpeedStamped>) ostream)
  "Serializes a message object of type '<JointSpeedStamped>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'speed) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <JointSpeedStamped>) istream)
  "Deserializes a message object of type '<JointSpeedStamped>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'speed) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<JointSpeedStamped>)))
  "Returns string type for a message object of type '<JointSpeedStamped>"
  "mobiman_msgs/JointSpeedStamped")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'JointSpeedStamped)))
  "Returns string type for a message object of type 'JointSpeedStamped"
  "mobiman_msgs/JointSpeedStamped")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<JointSpeedStamped>)))
  "Returns md5sum for a message object of type '<JointSpeedStamped>"
  "1627a98ed9651b259c8deabb0e0965fd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'JointSpeedStamped)))
  "Returns md5sum for a message object of type 'JointSpeedStamped"
  "1627a98ed9651b259c8deabb0e0965fd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<JointSpeedStamped>)))
  "Returns full string definition for message of type '<JointSpeedStamped>"
  (cl:format cl:nil "Header header~%JointSpeed speed~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: mobiman_msgs/JointSpeed~%float32[] speed~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'JointSpeedStamped)))
  "Returns full string definition for message of type 'JointSpeedStamped"
  (cl:format cl:nil "Header header~%JointSpeed speed~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: mobiman_msgs/JointSpeed~%float32[] speed~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <JointSpeedStamped>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'speed))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <JointSpeedStamped>))
  "Converts a ROS message object to a list"
  (cl:list 'JointSpeedStamped
    (cl:cons ':header (header msg))
    (cl:cons ':speed (speed msg))
))
