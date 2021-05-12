#pragma once
template<typename T>
class DifferentiableFunction{
public:
	virtual void const get_parameters(T** p) = 0;
	virtual T const evaluate(const T x) = 0;
	virtual T const derivative(const T x) = 0;
	virtual T const derivative(const T x, T* p) = 0;
	virtual bool const depends_directly_on(T* p) = 0;
	virtual uint32_t const get_number_parameters() = 0;
};
