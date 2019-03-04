inline float exp1(float x) {
	x = 1.0 + x / 256.0;
	x *= x; x *= x; x *= x; x *= x;
	x *= x; x *= x; x *= x; x *= x;
	return x;
}

inline float exp2(double x) { 
	x = 1.0 + x / 1024;   
	x *= x; x *= x; x *= x; x *= x;   
	x *= x; x *= x; x *= x; x *= x;   
	x *= x; x *= x;   
	return x; 
}