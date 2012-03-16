#ifndef RAYTRACING
#define RAYTRACING

#include "ExampleBase.h"
#include <string>

class RayTracing : public ExampleBase {

    class Sphere {

	public:
        float r;		//, g, b;
        float radius;
        float x, y, z;

        const float hit(const float ox, const float oy, float &n) const {
            float
				dx = ox - this->x,
				dx2 = dx * dx,
				dy = oy - this->y,
				dy2 = dy * dy,
				radius2 = this->radius * this->radius;

            if (dx2 + dy2 < radius2) {
                float dz = sqrt(radius2 - dx2 - dy2);
                n = dz / sqrt(radius2);
                return dz + this->z;
            }

            return FLT_MIN;
        }
    };

public:

	RayTracing();

	~RayTracing();

	virtual std::string getName() const;


protected:

    cl_mem bitmap;
	cl_mem memSizeX;
	cl_mem memSizeY;
	cl_mem memSizeZ;
	cl_mem spheresR;
	cl_mem spheresRadius;
	cl_mem spheresX;
	cl_mem spheresY;
	cl_mem spheresZ;
	Sphere *spheres;

	void scaleAndSetSizes(float sizeX, float sizeY, float sizeZ);

	void discardMembers();

	void initializeMembers();

	void performAlgorithm();

	const char *algorithm() const;

	const bool isValid() const;
};

#endif