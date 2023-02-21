OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
rz(1*pi) q[1];
h q[2];
rz(1*pi) q[2];
h q[3];
rz(0.5*pi) q[3];
cz q[3], q[0];
cz q[3], q[1];
cz q[3], q[2];
h q[0];
h q[1];
h q[2];
