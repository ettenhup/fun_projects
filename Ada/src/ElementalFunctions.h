#include <cstdint>
#include <vector>
#include "DifferentiableFunction.h"

template<typename T>
class ElementalFunction {
public:
	enum EnumDiff { is_elemental = true };
	static constexpr uint32_t get_static_number_parameters(){
		return 0;
	};
	static DifferentiableFunction<T>* get_instance(T* p){
		return nullptr;
	}
};
template<typename T>
class ReLu: public DifferentiableFunction<T>, public ElementalFunction<T> {
public:
	static constexpr uint32_t get_static_number_parameters(){
		return 0;
	}
	virtual uint32_t const get_number_parameters(){
		return get_static_number_parameters();
	}
	static DifferentiableFunction<T>* get_instance(T* p){
		return new ReLu<T>();
	}
	virtual void const get_parameters(T** p){
	}
	virtual T const evaluate(const T x){
		return x > 0 ? x : static_cast<T>(0);
	}
	virtual T const derivative(const T x){
		return x > 0 ?static_cast<T>(1) : static_cast<T>(0);
	}
	virtual T const derivative(const T x, T* p){
		return static_cast<T>(0);
	}
	virtual bool const depends_directly_on(T* p) {
		return false;
	}
};

template<typename T>
class Exp: public DifferentiableFunction<T>, public ElementalFunction<T> {
public:
	static constexpr uint32_t get_static_number_parameters(){
		return 0;
	}
	virtual uint32_t const get_number_parameters(){
		return get_static_number_parameters();
	}
	static DifferentiableFunction<T>* get_instance(T* p){
		return new Exp<T>();
	}
	virtual void const get_parameters(T** p){
	}
	virtual T const evaluate(const T x){
		return exp(x);
	}
	virtual T const derivative(const T x){
		return exp(x);
	}
	virtual T const derivative(const T x, T* p){
		return static_cast<T>(0);
	}
	virtual bool const depends_directly_on(T* p) {
		return false;
	}
};

template<typename T>
class Linear: public DifferentiableFunction<T>, public ElementalFunction<T> {
	T *a, *b;
public:
	Linear(T* a, T* b) : a(a), b(b){};
	Linear(T* a) : a(a), b(a+1){};
	static DifferentiableFunction<T>* get_instance(T* p){
		return new Linear<T>(p);
	}
	static constexpr uint32_t get_static_number_parameters(){
		return 2;
	}
	virtual uint32_t const get_number_parameters(){
		return get_static_number_parameters();
	}
	virtual void const get_parameters(T** p){
		p[0] = a;
		p[1] = b;
	}
	virtual T const evaluate(const T x){
		return (*a) * x + *b; 
	}
	virtual T const derivative(const T x){
		return *a;
	}
	virtual T const derivative(const T x,T* p){
		if(p == a){
			return x;
		}
		else if(p==b){
			return static_cast<T>(1);
		}
		return static_cast<T>(0);
	}
	virtual bool const depends_directly_on(T* p) {
		return p==a || p==b;
	}
};

template<typename T>
class GaussCurve: public DifferentiableFunction<T>, public ElementalFunction<T> {
	T *a, *b, *c;
public:
	GaussCurve(T* a, T* b, T* c) : a(a), b(b), c(c){};
	GaussCurve(T* a) : a(a), b(a+1), c(a+2){};
	static DifferentiableFunction<T>* get_instance(T* p){
		return new GaussCurve<T>(p);
	}
	static constexpr uint32_t get_static_number_parameters(){
		return 3;
	}
	virtual uint32_t const get_number_parameters(){
		return get_static_number_parameters();
	}
	virtual void const get_parameters(T** p){
		p[0] = a;
		p[1] = b;
		p[2] = c;
	}
	virtual T const evaluate(const T x){
		const T d = x - (*c);
		return (*a) * exp((*b) * d * d); 
	}
	virtual T const derivative(const T x){
		return 2*(*b)*(x-(*c))*evaluate(x);
	}
	virtual T const derivative(const T x,T* p){
		if(p == a){
			return evaluate(x) / (*a);
		}
		else if(p==b){
			const T d = x - (*c);
			return d*d*evaluate(x);
		}
		else if(p==c){
			return -2*(*b)*(x-(*c))*evaluate(x);
		}
		return static_cast<T>(0);
	}
	virtual bool const depends_directly_on(T* p) {
		return p==a || p==b || p==c;
	}
};
