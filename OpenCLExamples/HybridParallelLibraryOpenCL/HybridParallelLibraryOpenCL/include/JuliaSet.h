#ifndef JULIASET
#define JULIASET

#include "ExampleBase.h"
#include <string>

class JuliaSet : public ExampleBase {
        class Complex {

		private:
            float r;
            float i;

		public:

            Complex(const float r, const float i) 
				:	r(r),
					i(i) { }

			~Complex() { }

            float magnitude2() {
                return this->r * this->r + this->i * this->i;
            }

            Complex operator*(const Complex &c) {
                return Complex(
					this->r * c.r - this->i * c.i,
					this->i * c.r + this->r * c.i);
            }

            Complex operator+(const Complex &c) {
                return Complex(
					this->r + c.r,
					this->i + c.i);
            }
		};

public:

	JuliaSet();

	~JuliaSet();

	virtual std::string getName() const;


protected:

    cl_mem bitmap;
	cl_mem memSizeX;
	cl_mem memSizeY;

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void performAlgorithm();

	const char *algorithm() const;

	const bool isValid() const;

	const size_t julia(const int x, const int y) const;
};

#endif