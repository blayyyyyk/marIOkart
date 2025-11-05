from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
from private.mkds_python_bindings.generated.gsPlatform import *

packet_map1a = 0
packet_map2 = 1
packet_map3 = 2
packet_map1b = 3
NUM_PACKETS = 4

packet_map1a = 0
packet_map2 = 1
packet_map3 = 2
packet_map1b = 3
NUM_PACKETS = 4

no_nat = 0
firewall_only = 1
full_cone = 2
restricted_cone = 3
port_restricted_cone = 4
symmetric = 5
unknown = 6
NUM_NAT_TYPES = 7

no_nat = 0
firewall_only = 1
full_cone = 2
restricted_cone = 3
port_restricted_cone = 4
symmetric = 5
unknown = 6
NUM_NAT_TYPES = 7

promiscuous = 0
not_promiscuous = 1
port_promiscuous = 2
ip_promiscuous = 3
promiscuity_not_applicable = 4
NUM_PROMISCUITY_TYPES = 5

promiscuous = 0
not_promiscuous = 1
port_promiscuous = 2
ip_promiscuous = 3
promiscuity_not_applicable = 4
NUM_PROMISCUITY_TYPES = 5

unrecognized = 0
private_as_public = 1
consistent_port = 2
incremental = 3
mixed = 4
NUM_MAPPING_SCHEMES = 5

unrecognized = 0
private_as_public = 1
consistent_port = 2
incremental = 3
mixed = 4
NUM_MAPPING_SCHEMES = 5

packet_map1a = 0
packet_map2 = 1
packet_map3 = 2
packet_map1b = 3
NUM_PACKETS = 4

packet_map1a = 0
packet_map2 = 1
packet_map3 = 2
packet_map1b = 3
NUM_PACKETS = 4

no_nat = 0
firewall_only = 1
full_cone = 2
restricted_cone = 3
port_restricted_cone = 4
symmetric = 5
unknown = 6
NUM_NAT_TYPES = 7

no_nat = 0
firewall_only = 1
full_cone = 2
restricted_cone = 3
port_restricted_cone = 4
symmetric = 5
unknown = 6
NUM_NAT_TYPES = 7

promiscuous = 0
not_promiscuous = 1
port_promiscuous = 2
ip_promiscuous = 3
promiscuity_not_applicable = 4
NUM_PROMISCUITY_TYPES = 5

promiscuous = 0
not_promiscuous = 1
port_promiscuous = 2
ip_promiscuous = 3
promiscuity_not_applicable = 4
NUM_PROMISCUITY_TYPES = 5

unrecognized = 0
private_as_public = 1
consistent_port = 2
incremental = 3
mixed = 4
NUM_MAPPING_SCHEMES = 5

unrecognized = 0
private_as_public = 1
consistent_port = 2
incremental = 3
mixed = 4
NUM_MAPPING_SCHEMES = 5

packet_map1a = 0
packet_map2 = 1
packet_map3 = 2
packet_map1b = 3
NUM_PACKETS = 4

packet_map1a = 0
packet_map2 = 1
packet_map3 = 2
packet_map1b = 3
NUM_PACKETS = 4

no_nat = 0
firewall_only = 1
full_cone = 2
restricted_cone = 3
port_restricted_cone = 4
symmetric = 5
unknown = 6
NUM_NAT_TYPES = 7

no_nat = 0
firewall_only = 1
full_cone = 2
restricted_cone = 3
port_restricted_cone = 4
symmetric = 5
unknown = 6
NUM_NAT_TYPES = 7

promiscuous = 0
not_promiscuous = 1
port_promiscuous = 2
ip_promiscuous = 3
promiscuity_not_applicable = 4
NUM_PROMISCUITY_TYPES = 5

promiscuous = 0
not_promiscuous = 1
port_promiscuous = 2
ip_promiscuous = 3
promiscuity_not_applicable = 4
NUM_PROMISCUITY_TYPES = 5

unrecognized = 0
private_as_public = 1
consistent_port = 2
incremental = 3
mixed = 4
NUM_MAPPING_SCHEMES = 5

unrecognized = 0
private_as_public = 1
consistent_port = 2
incremental = 3
mixed = 4
NUM_MAPPING_SCHEMES = 5


class _AddressMapping(Structure):
    _fields_ = [
        ('privateIp', c_int),
        ('privatePort', c_short),
        ('publicIp', c_int),
        ('publicPort', c_short),
    ]

class _NAT(Structure):
    _fields_ = [
        ('brand', (c_char * 32)),
        ('model', (c_char * 32)),
        ('firmware', (c_char * 64)),
        ('ipRestricted', gsi_bool),
        ('portRestricted', gsi_bool),
        ('promiscuity', NatPromiscuity),
        ('natType', NatType),
        ('mappingScheme', NatMappingScheme),
        ('mappings', (AddressMapping * 4)),
        ('qr2Compatible', gsi_bool),
    ]
NatifyPacket = c_int
NatPromiscuity = c_int
NatType = c_int
NatMappingScheme = c_int
