/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <sstream>

#include "logarithmic_mapping.hpp"
#include "collapsing_highest_dense_store.hpp"
#include "ddsketch.hpp"

namespace nb = nanobind;

template<class S, class M>
void bind_ddsketch(nb::module_ &m, const char* name) {
  using namespace datasketches;

  using RetStr = datasketches::string<std::allocator<char>>;
  auto ddsketch_class = nb::class_<DDSketch<S, M>>(m, name)
    .def(nb::init<double>(), nb::arg("relative_accuracy") = 0.01,
         "Creates a ddsketch instance with the given relative accuracy.\n\n"
         ":param relative_accuracy: sets the relative accuracy.\n"
         ":type relative_accuracy: float, optional. Default = 0.01")
    .def("__copy__", [](const DDSketch<S, M>& other) { return DDSketch<S, M>(other); })
    .def("update", &DDSketch<S, M>::update, nb::arg("value"), nb::arg("count") = 1.0,
         "Updates the sketch with the given value and optional count")
    .def("merge",
         (void (DDSketch<S, M>::*)(const DDSketch<S, M>&)) &DDSketch<S, M>::merge,
         nb::arg("sketch"),
         "Merges another sketch of the same type into this one")
    .def("get_rank", &DDSketch<S, M>::get_rank, nb::arg("item"),
         "Returns the approximate normalized rank of the given item")
    .def("get_quantile", (double (DDSketch<S, M>::*)(const double&) const) &DDSketch<S, M>::get_quantile,
         nb::arg("rank"),
         "Returns an approximation to the data value associated with the given rank")
    .def("is_empty", &DDSketch<S, M>::is_empty)
    .def("clear", &DDSketch<S, M>::clear)
    .def("get_count", &DDSketch<S, M>::get_count)
    .def("get_sum", &DDSketch<S, M>::get_sum)
    .def("get_min", &DDSketch<S, M>::get_min)
    .def("get_max", &DDSketch<S, M>::get_max)
    .def("to_string", static_cast<RetStr (DDSketch<S, M>::*)() const>( &DDSketch<S, M>::to_string),
      "Produces a string summary of the sketch")
    .def("__str__", static_cast<RetStr (DDSketch<S, M>::*)() const>(&DDSketch<S, M>::to_string),
         "Produces a string summary of the sketch")
    .def("get_serialized_size_bytes", [](const DDSketch<S, M>& sk) {
        std::ostringstream os;
        sk.serialize(os);
        const std::string& s = os.str();
        return static_cast<size_t>(s.size());
      },
      "Returns the size of the serialized sketch, in bytes")
    .def_static("deserialize", [](nb::bytes b) {
        // Copy bytes into a std::string so the storage outlives the stream
        std::string s(b.c_str(), b.size());
        std::istringstream is(s);
        return DDSketch<S, M>::deserialize(is);
      }, nb::arg("data"),
      "Deserialize a sketch from Python bytes")
    .def("serialize", [](const DDSketch<S, M>& sk) {
        std::ostringstream os;
        sk.serialize(os);
        const std::string& s = os.str();
        return nb::bytes(s.data(), s.size());
      },
      "Serialize a sketch from Python bytes")
  ;


}

void init_ddsketch(nb::module_ &m) {
  // bind_ddsketch<datasketches::SparseStore<std::allocator<double>>, datasketches::LogarithmicMapping>(m, "ddsketch");
  bind_ddsketch<datasketches::CollapsingHighestDenseStore<1024, std::allocator<double>>, datasketches::LogarithmicMapping>(m, "ddsketch_dense");
}