OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
rz(1*pi) q[4];
cz q[2], q[0];
cz q[3], q[0];
cz q[3], q[2];
cz q[4], q[0];
cz q[4], q[1];
cz q[4], q[2];
cz q[4], q[3];
h q[4];
