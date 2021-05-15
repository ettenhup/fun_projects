#pragma once
#include <cstdint>
#include <initializer_list>
#include <vector>
#include "DifferentiableFunction.h"

template<typename T>
class ElementalFunction {
public:
	enum EnumDiff { is_elemental = true };
	static constexpr uint32_t get_static_number_parameters(){
		return 0;
	};
	static DifferentiableFunction<T>* get_instance(){
		return nullptr;
	}
};
template<typename T>
class ElementalDerivedFunction {
public:
	enum EnumDiff { is_elemental = true };
	static constexpr uint32_t get_static_number_parameters(){
		return 0;
	};
	static DifferentiableFunction<T>* get_instance(){
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
	static DifferentiableFunction<T>* get_instance(){
		return new ReLu<T>();
	}
	virtual std::vector<T*> const get_parameters(){
	}
	virtual T const evaluate(const T x){
		return x > 0 ? x : static_cast<T>(0);
	}
	virtual T const derivative(const T x){
		return x > 0 ? static_cast<T>(1) : static_cast<T>(0);
	}
	virtual T const derivative(const T x, T* p){
		return static_cast<T>(0);
	}
	virtual bool const depends_directly_on(T* p) {
		return false;
	}
	virtual void align_parameters(T *p){
	}
	virtual void align_parameters(std::vector<T*> p){
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
	static DifferentiableFunction<T>* get_instance(){
		return new Exp<T>();
	}
	virtual std::vector<T*> const get_parameters(){
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
	virtual void align_parameters(T *p){
	}
	virtual void align_parameters(std::vector<T*> p){
	}
};

template<typename T>
class Linear: public DifferentiableFunction<T>, public ElementalDerivedFunction<T> {
	T *a, *b;
public:
	Linear(): a(nullptr), b(nullptr){};
	static DifferentiableFunction<T>* get_instance(){
		return new Linear<T>();
	}
	static constexpr uint32_t get_static_number_parameters(){
		return 2;
	}
	virtual uint32_t const get_number_parameters(){
		return get_static_number_parameters();
	}
	virtual std::vector<T*> const get_parameters(){
		assert(a);
		assert(b);
		return std::vector<T*>{a,b};
	}
	virtual T const evaluate(const T x){
		assert(a);
		assert(b);
		return (*a) * x + *b; 
	}
	virtual T const derivative(const T x){
		assert(a);
		assert(b);
		return *a;
	}
	virtual T const derivative(const T x,T* p){
		assert(a);
		assert(b);
		if(p == a){
			return x;
		}
		else if(p==b){
			return static_cast<T>(1);
		}
		return static_cast<T>(0);
	}
	virtual bool const depends_directly_on(T* p) {
		assert(a);
		assert(b);
		return p==a || p==b;
	}
	virtual void align_parameters(T *p){
		this->align_external_to_internal_parameter(&a,p);
		this->align_external_to_internal_parameter(&b,p+1);
	}
	virtual void align_parameters(std::vector<T*> p){
		assert(p.size()==get_static_number_parameters());
		this->align_external_to_internal_parameter(&a,*(p.begin()));
		this->align_external_to_internal_parameter(&b,*(p.begin()+1));
	}
};

template<typename T>
class GaussCurve: public DifferentiableFunction<T>, public ElementalDerivedFunction<T> {
	T *a, *b, *c;
public:
	GaussCurve() : a(nullptr), b(nullptr), c(nullptr){};
	static DifferentiableFunction<T>* get_instance(){
		return new GaussCurve<T>();
	}
	static constexpr uint32_t get_static_number_parameters(){
		return 3;
	}
	virtual uint32_t const get_number_parameters(){
		return get_static_number_parameters();
	}
	virtual std::vector<T*> const get_parameters(){
		assert(a);
		assert(b);
		assert(c);
		return std::vector<T*>{a,b,c};
	}
	virtual T const evaluate(const T x){
		assert(a);
		assert(b);
		assert(c);
		const T d = x-(*c);
		return (*a) * exp(-(*b) * d * d); 
	}
	virtual T const derivative(const T x){
		assert(a);
		assert(b);
		assert(c);
		return -2*(*b)*(x-(*c))*evaluate(x);
	}
	virtual T const derivative(const T x,T* p){
		assert(a);
		assert(b);
		assert(c);
		if(p == a){
			return evaluate(x) / (*a);
		}
		else if(p==b){
			const T d = x-(*c);
			return -d*d*evaluate(x);
		}
		else if(p==c){
			return 2*(*b)*(x-(*c))*evaluate(x);
		}
		return static_cast<T>(0);
	}
	virtual bool const depends_directly_on(T* p) {
		assert(a);
		assert(b);
		assert(c);
		return p==a || p==b || p==c;
	}
	virtual void align_parameters(T *p){
		this->align_external_to_internal_parameter(&a,p);
		this->align_external_to_internal_parameter(&b,p+1);
		this->align_external_to_internal_parameter(&c,p+2);
	}
	virtual void align_parameters(std::vector<T*> p){
		assert(p.size()==get_static_number_parameters());
		this->align_external_to_internal_parameter(&a,*(p.begin()));
		this->align_external_to_internal_parameter(&b,*(p.begin()+1));
		this->align_external_to_internal_parameter(&c,*(p.begin()+2));
	}
};
