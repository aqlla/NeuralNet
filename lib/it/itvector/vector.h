#ifndef __it_VEC_h__
#define __it_VEC_h__

#include "ittypes.h"
#include <iostream>
#include <sstream>
#include "math.h"

namespace it {

    template<typename T>
    struct vec3
    {
        T arr[3];
        T &x = arr[0];
        T &y = arr[1];
        T &z = arr[2];

        // Constructors
        vec3(vec3<T> &vec3);

        explicit vec3(T x = 0, T y = 0, T z = 0);

        explicit vec3(T *arr);

        // Destructor
        virtual ~vec3() = default;

        void set(T x, T y, T z);

        // Math functions
        double magnitude() const;

        vec3<T> getNormalized() const;

        vec3<T> &normalize();

        vec3<T> cross(const vec3<T> &rhs) const;

        T dot(const vec3<T> &rhs) const;

        int sign(const vec3<T> &rhs) const;

        /*   Assignment Operator   */
        virtual vec3<T> operator=(const vec3<T> &rhs);

        /*   Arithmetic Operators   */
        vec3<T> operator+(const vec3<T> &rhs) const;

        vec3<T> operator+(const T rhs) const;

        vec3<T> operator-(const vec3<T> &rhs) const;

        vec3<T> operator-(const T rhs) const;

        vec3<T> operator/(const vec3<T> &rhs) const;

        vec3<T> operator*(const vec3<T> &rhs) const;

        vec3<T> operator*(const T rhs) const;

        vec3<T> operator/(const T rhs) const;

        /*   Arithmetic & Assignment Operators   */
        vec3<T> &operator+=(const vec3<T> &rhs);

        vec3<T> &operator+=(const T rhs);

        vec3<T> &operator-=(const vec3<T> &rhs);

        vec3<T> &operator-=(const T rhs);

        vec3<T> &operator*=(const vec3<T> &rhs);

        vec3<T> &operator*=(const T rhs);

        vec3<T> &operator/=(const vec3<T> &rhs);

        vec3<T> &operator/=(const T rhs);

        /*   Reference Operators   */
        operator T *() const;

        T &operator[](int index);

        const T &operator[](int index) const;

        std::string to_string() const;
    };


    template<typename T>
    vec3<T>::vec3(vec3<T> &vec3)
            : arr{vec3.x, vec3.y, vec3.z} { };

    template<typename T>
    vec3<T>::vec3(T x, T y, T z)
            : arr{x, y, z} { };

    template<typename T>
    vec3<T>::vec3(T *arr)
            : arr{arr[0], arr[1], arr[2]} { };

    template<typename T>
    void vec3<T>::set(T x, T y, T z) {
        this->x = x;
        this->y = y;
        this->z = z;
    };

// Math functions
    template<typename T>
    double vec3<T>::magnitude() const {
        return sqrt(x * x + y * y + z * z);
    };

    template<typename T>
    vec3<T> vec3<T>::getNormalized() const {
        return vec3<T>(x, y, z).normalize();
    };

    template<typename T>
    vec3<T> &vec3<T>::normalize() {
        double magnitude = this->magnitude();
        x /= magnitude;
        y /= magnitude;
        z /= magnitude;
        return *this;
    };

    template<typename T>
    T vec3<T>::dot(const vec3<T> &rhs) const {
        return x * rhs.x + y * rhs.y + z * rhs.z;
    };

    template<typename T>
    vec3<T> vec3<T>::cross(const vec3<T> &rhs) const {
        return vec3<T>(
                (y * rhs.z) - (z * rhs.y),
                (z * rhs.x) - (x * rhs.z),
                (x * rhs.y) - (y * rhs.x)
        );
    };

    template<typename T>
    int vec3<T>::sign(const vec3<T> &rhs) const {
        if (x * rhs.x > x * rhs.y) {
            return 1;
        } else {
            return -1;
        }
    };

    template<typename T>
    vec3<T> vec3<T>::operator=(const vec3<T> &rhs) {
        if (this != &rhs) {
            x = rhs.x;
            y = rhs.y;
            z = rhs.z;
        }

        return *this;
    }

/*   Arithmetic Operators   */
    template<typename T>
    vec3<T> vec3<T>::operator+(const vec3<T> &rhs) const {
        return vec3<T>(x + rhs.x, y + rhs.y, z + rhs.z);
    };

    template<typename T>
    vec3<T> vec3<T>::operator+(const T rhs) const {
        return vec3<T>(x + rhs, y + rhs, z + rhs);
    };

    template<typename T>
    vec3<T> vec3<T>::operator-(const vec3<T> &rhs) const {
        return vec3<T>(x - rhs.x, y - rhs.y, z - rhs.z);
    };

    template<typename T>
    vec3<T> vec3<T>::operator-(const T rhs) const {
        return vec3<T>(x - rhs, y - rhs, z - rhs);
    };

    template<typename T>
    inline vec3<T> operator-(const T lhs, const vec3<T> &rhs) {
        return vec3<T>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
    };

    template<typename T>
    inline vec3<T> operator-(const vec3<T> &lhs, const T rhs) {
        return vec3<T>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
    };

    template<typename T>
    vec3<T> vec3<T>::operator/(const vec3<T> &rhs) const {
        return vec3<T>(x / rhs.x, y / rhs.y, z / rhs.z);
    };

    template<typename T>
    vec3<T> vec3<T>::operator*(const vec3<T> &rhs) const {
        return vec3<T>(x * rhs.x, y * rhs.y, z * rhs.z);
    };

    template<typename T>
    vec3<T> vec3<T>::operator*(const T rhs) const {
        return vec3<T>(x * rhs, y * rhs, z * rhs);
    };

    template<typename T>
    vec3<T> vec3<T>::operator/(const T rhs) const {
        return vec3<T>(x / (double) rhs, y / (double) rhs, z / (double) rhs);
    };


/*   Arithmetic & Assignment Operators   */
    template<typename T>
    vec3<T> &vec3<T>::operator+=(const vec3<T> &rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    template<typename T>
    vec3<T> &vec3<T>::operator+=(const T rhs) {
        x += rhs;
        y += rhs;
        z += rhs;
        return *this;
    }

    template<typename T>
    vec3<T> &vec3<T>::operator-=(const vec3<T> &rhs) {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    template<typename T>
    vec3<T> &vec3<T>::operator-=(const T rhs) {
        x -= rhs;
        y -= rhs;
        z -= rhs;
        return *this;
    }

    template<typename T>
    vec3<T> &vec3<T>::operator*=(const vec3<T> &rhs) {
        x *= rhs.x;
        y *= rhs.y;
        z *= rhs.z;
        return *this;
    }

    template<typename T>
    vec3<T> &vec3<T>::operator*=(const T rhs) {
        x *= rhs;
        y *= rhs;
        z *= rhs;
        return *this;
    }

    template<typename T>
    vec3<T> &vec3<T>::operator/=(const vec3<T> &rhs) {
        x /= rhs.x;
        y /= rhs.y;
        z /= rhs.z;
        return *this;
    }

    template<typename T>
    vec3<T> &vec3<T>::operator/=(const T rhs) {
        x /= rhs;
        y /= rhs;
        z /= rhs;
        return *this;
    }


/*   Reference Operators   */
    template<typename T>
    T &vec3<T>::operator[](int index) {
        switch (index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: return x;
        }
    }

    template<typename T>
    const T &vec3<T>::operator[](int index) const {
        switch (index) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: return x;
        }
    }

    template<typename T>
    vec3<T>::operator T *() const {
        return &arr[0];
    }

    template<typename T>
    inline std::ostream &operator<<(std::ostream &out, const vec3<T> &vec) {
        return out << (vec.to_string());
    }

    template<typename T>
    std::string vec3<T>::to_string() const {
        std::stringstream ss;
        ss << "[" << x << ", " << y << ", " << z << "]";
        return ss.str();
    }


};

#endif