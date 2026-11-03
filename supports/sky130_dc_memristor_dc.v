// Copyright 2020 The SkyWater PDK Authors
// DC mode
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//
// DC-only variant of sky130_dc_memristor for SPICECiM-X inference mode.
// ---------------------------------------------------------------------
// This model is intentionally stateless in time:
//   - No filament thickness update (no `Tfilament_dTdt`)
//   - No thermal dynamics (temperature fixed at `Temperature_0`)
//   - No `@(initial_step)` / `$abstime` / `t_delta` machinery
//   - Filament thickness is taken DIRECTLY from the `Tfilament_init` parameter
//
// The I-V relationship is preserved exactly from the original model so that
// DC-mode results match transient-mode steady-state results bit-for-bit at
// the same operating point.
//
// Conductance is controlled at instantiation time by setting `Tfilament_init`.
// Per the original model:
//     G(V=0+) ~ (I_k1 / V_ref) * exp(-(Tox - Tfilament_init)/(Tox - Tfilament_ref))
// so larger `Tfilament_init` -> larger conductance.
//
// This file is designed for use with SPICE DC operating-point (.op) and
// DC sweep (.dc) analyses. It also works correctly in transient analysis
// (it will simply behave as a static nonlinear resistor with no memory).
// ---------------------------------------------------------------------

`include "constants.vams"
`include "disciplines.vams"

module sky130_dc_memristor_dc(TE, BE);
    inout TE;                    // top electrode
    inout BE;                    // bottom electrode
    electrical TE, BE;

    // ==================================================
    // Physical / geometric parameters (kept for parity with the dynamic model)
    // --------------------------------------------------
    parameter real area_ox        = 0.1024e-12 from (0:inf); // TE/BE overlap area [m^2]
    parameter real Tox            = 5.0e-9     from (0:inf); // oxide thickness [m]
    parameter real Tfilament_max  = 4.9e-9     from (0:inf); // max filament thickness (LRS bound) [m]
    parameter real Tfilament_min  = 3.3e-9     from (0:inf); // min filament thickness (HRS bound) [m]

    // ==================================================
    // CONDUCTANCE CONTROL KNOB
    // --------------------------------------------------
    // This is THE parameter you sweep to program the cell.
    // Bounded to the physical filament range so out-of-range writes
    // are caught at instantiation time rather than producing garbage I-V.
    parameter real Tfilament_init = 3.3e-9 from [Tfilament_min : Tfilament_max];

    // ==================================================
    // I-V calibration parameters (identical to dynamic model)
    // --------------------------------------------------
    parameter real I_k1           = 6.140e-5  from (0:inf); // current scale [A]
    parameter real Tfilament_ref  = 4.7249e-9 from (0:inf); // filament thickness reference [m]
    parameter real V_ref          = 0.430     from (0:inf); // voltage scale [V]

    // ==================================================
    // Temperature (fixed in DC mode; no self-heating)
    // --------------------------------------------------
    // Exposed so PVT corner sweeps can still set ambient temperature.
    parameter real Temperature_0  = 300       from (0:inf); // ambient temperature [K]

    // ==================================================
    // Core equation
    // --------------------------------------------------
    // I = I_k1 * exp(-(Tox - Tfilament_init)/(Tox - Tfilament_ref)) * sinh(V/V_ref)
    //
    // This is identical to the steady-state form of the original model when
    // `Tfilament_current == Tfilament_init` and self-heating is neglected.
    // It is purely instantaneous (no ddt, no internal state), so SPICE DC
    // analysis solves it in a single Newton iteration per operating point.
    // ==================================================
    analog begin
        I(TE,BE) <+ I_k1
                  * exp(-(Tox - Tfilament_init) / (Tox - Tfilament_ref))
                  * sinh( V(TE,BE) / V_ref );
    end

endmodule
