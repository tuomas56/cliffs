OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
h q[0];
rz(0.5*pi) q[0];
h q[1];
h q[2];
rz(0.5*pi) q[2];
h q[3];
rz(0.5*pi) q[3];
h q[4];