from ctypes import *
from private.mkds_python_bindings.pointer import POINTER32, c_void_p32
ns_initsent = 0
ns_initack = 1
ns_connectping = 2
ns_finished = 3
ns_canceled = 4
ns_reportsent = 5
ns_reportack = 6

ns_initsent = 0
ns_initack = 1
ns_connectping = 2
ns_finished = 3
ns_canceled = 4
ns_reportsent = 5
ns_reportack = 6

nr_success = 0
nr_deadbeatpartner = 1
nr_inittimeout = 2
nr_pingtimeout = 3
nr_unknownerror = 4
nr_noresult = 5

nr_success = 0
nr_deadbeatpartner = 1
nr_inittimeout = 2
nr_pingtimeout = 3
nr_unknownerror = 4
nr_noresult = 5

ne_noerror = 0
ne_allocerror = 1
ne_socketerror = 2
ne_dnserror = 3

ne_noerror = 0
ne_allocerror = 1
ne_socketerror = 2
ne_dnserror = 3

ns_initsent = 0
ns_initack = 1
ns_connectping = 2
ns_finished = 3
ns_canceled = 4
ns_reportsent = 5
ns_reportack = 6

ns_initsent = 0
ns_initack = 1
ns_connectping = 2
ns_finished = 3
ns_canceled = 4
ns_reportsent = 5
ns_reportack = 6

nr_success = 0
nr_deadbeatpartner = 1
nr_inittimeout = 2
nr_pingtimeout = 3
nr_unknownerror = 4
nr_noresult = 5

nr_success = 0
nr_deadbeatpartner = 1
nr_inittimeout = 2
nr_pingtimeout = 3
nr_unknownerror = 4
nr_noresult = 5

ne_noerror = 0
ne_allocerror = 1
ne_socketerror = 2
ne_dnserror = 3

ne_noerror = 0
ne_allocerror = 1
ne_socketerror = 2
ne_dnserror = 3

NegotiateState = c_int
NegotiateProgressFunc = c_void_p32
NegotiateCompletedFunc = c_void_p32
NegotiateResult = c_int
NatDetectionResultsFunc = c_void_p32
NegotiateError = c_int
