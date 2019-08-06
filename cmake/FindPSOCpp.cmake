# FindPSOCpp.txt
#
#     Author: Fabian Meyer
# Created On: 05 Aug 2019
#
# Defines
#   PSOCPP_INCLUDE_DIR
#   PSOCPP_FOUND

find_path(PSOCPP_INCLUDE_DIR "psocpp.h"
    HINTS
    "${PSOCPP_ROOT}/include"
    "$ENV{PSOCPP_ROOT}/include"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PSOCPP DEFAULT_MSG PSOCPP_INCLUDE_DIR)
