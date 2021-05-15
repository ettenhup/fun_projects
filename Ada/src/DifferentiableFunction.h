#pragma once
#include <vector>
template<typename T>
class ParametrizedFunction{
public:
	virtual std::vector<T*> const get_parameters() = 0;
	virtual bool const depends_directly_on(T* p) = 0;
	virtual uint32_t const get_number_parameters() = 0;
	virtual void align_parameters(T* p) = 0;
	virtual void align_parameters(std::vector<T*> p) = 0;
	//virtual bool const parameters_aligned() = 0;
	//
	void align_external_to_internal_parameter(T** internal, T *p){
		bool transferValue = *internal != nullptr;
		T tmp;
		if(transferValue){
			tmp = **internal;
		}
		*internal = p;
		if(transferValue){
			**internal = tmp;
		}
	}
};

template<typename T>
class DifferentiableFunction : virtual public ParametrizedFunction<T>{
public:
	virtual T const evaluate(const T x) = 0;
	virtual T const derivative(const T x) = 0;
	virtual T const derivative(const T x, T* p) = 0;
};
