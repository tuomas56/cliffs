OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
h q[1];
rz(1*pi) q[1];
cz q[1], q[0];
h q[1];
