// Optional pybind11 module for building _paintcore Python extension via scikit-build-core / CMake
#if defined(PAINTCORE_WITH_PYBIND11)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "paintcore/types.hpp"
#include "paintcore/color_spaces.hpp"
#include "paintcore/km_brdf_solver.hpp"

namespace py = pybind11;
using namespace paintcore;

PYBIND11_MODULE(_paintcore, m) {
    m.doc() = "C++20 Native Colorimetry & Kubelka-Munk Physics Core (libpaintcore)";

    m.def("evaluate_km_channel", &physics::evaluate_km_channel,
          py::arg("R_prev"), py::arg("a"), py::arg("b"), py::arg("SX"),
          py::call_guard<py::gil_scoped_release>());

    m.def("delta_e00_linear_rgb", &color::delta_e00_linear_rgb,
          py::arg("rgb1"), py::arg("rgb2"),
          py::call_guard<py::gil_scoped_release>());

    m.def("verify_zero_copy_ptr", [](py::array_t<float, py::array::c_style> arr) -> uintptr_t {
        py::buffer_info info = arr.request();
        return reinterpret_cast<uintptr_t>(info.ptr);
    });
}
#endif
