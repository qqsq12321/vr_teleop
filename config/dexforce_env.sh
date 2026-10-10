# DexForce W1 real-robot ROS 2 environment.
#
#   source config/dexforce_env.sh
#
# Machine side: this laptop reaches the robot over the wired LAN0 link
# (192.168.20.77 -> 192.168.20.20). The robot's driver binds CycloneDDS to its
# LAN0 address only, so this side must bind the same subnet or discovery fails
# silently and commands appear to be ignored.
#
# Adjust ADDRESS if this laptop's 192.168.20.x address changes:
#   ip -br addr show enp5s0

_ADDRESS="${DEXFORCE_DDS_ADDRESS:-192.168.20.77}"

export ROS_DOMAIN_ID=20
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# No <SharedMemory> section on purpose. The robot's deployment enables iceoryx
# shared memory, but RouDi only runs on the robot; enabling it here makes
# CycloneDDS print "RouDi not found - waiting ..." and stall discovery.
export CYCLONEDDS_URI="<CycloneDDS><Domain id=\"any\"><General><Interfaces><NetworkInterface address=\"${_ADDRESS}\" priority=\"100\" presence_required=\"false\" multicast=\"default\"/></Interfaces><AllowMulticast>default</AllowMulticast><DontRoute>false</DontRoute><RedundantNetworking>false</RedundantNetworking></General><Discovery><ParticipantIndex>auto</ParticipantIndex><LeaseDuration>10s</LeaseDuration><SPDPInterval>2s</SPDPInterval></Discovery></Domain></CycloneDDS>"

source /opt/ros/humble/setup.bash
source "$HOME/dexforce_msgs_ws/install/setup.bash"

# Conda's _ssl.so needs conda's libcrypto (OpenSSL 3.5); the system one is
# 3.0.x and lacks the OPENSSL_3.3.0 symbol it was linked against. The extension
# carries an $ORIGIN-relative RPATH which normally wins, but once any earlier
# library has pulled in /lib/x86_64-linux-gnu/libcrypto.so.3 the dynamic linker
# reuses it (same soname is loaded only once) and `import ssl` fails. Putting
# conda's lib first makes that impossible. Harmless when it is not needed.
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${CONDA_PREFIX}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

# ~/.bashrc exports ROS_DOMAIN_ID=25 for other projects; the robot runs on 20.
# Fail loudly rather than discovering it as "commands silently do nothing".
if [ "${ROS_DOMAIN_ID}" != "20" ]; then
  echo "WARNING: ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}, expected 20 (the robot's domain)" >&2
fi
if [ "${RMW_IMPLEMENTATION}" != "rmw_cyclonedds_cpp" ]; then
  echo "WARNING: RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}, expected rmw_cyclonedds_cpp" >&2
fi

echo "DexForce env: domain=${ROS_DOMAIN_ID} rmw=${RMW_IMPLEMENTATION} iface=${_ADDRESS}"
