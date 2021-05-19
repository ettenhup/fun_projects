#include <cstdint>
#include <cstdarg>
#include <random>
#include <iostream>
#include "DifferentiableFunction.h"
#pragma once
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0,1.0);

template<typename T>
class CombinationOfElementalFunctions: public virtual DifferentiableFunction<T> {
public:
	virtual std::vector<DifferentiableFunction<T>*>* const get_sub_functions() = 0;

	virtual uint32_t const get_number_parameters(){
		std::vector<DifferentiableFunction<T>*>* fs = get_sub_functions();
		return std::accumulate(fs->begin(), fs->end(), 0, [](uint32_t acc, DifferentiableFunction<T>* f){
				return acc + f->get_number_parameters();
			});
	}
	virtual std::vector<T*> const get_parameters(){
		std::vector<T*> allParam;
		std::vector<DifferentiableFunction<T>*>* fs = get_sub_functions();
		for(DifferentiableFunction<T>* f : *fs){
			std::vector<T*> thisFunParam = f->get_parameters();
			allParam.insert(allParam.end(), thisFunParam.begin(), thisFunParam.end());
		}
		return allParam;
	}
	virtual bool const depends_directly_on(T* p) {
		std::vector<DifferentiableFunction<T>*>* fs = get_sub_functions();
		return std::any_of(fs->begin(), fs->end(), [&p](DifferentiableFunction<T>* f){return f->depends_directly_on(p);});
	}
	virtual void align_parameters(T* p){
		std::vector<DifferentiableFunction<T>*>* fs = get_sub_functions();
		uint32_t offset = 0;
		for(DifferentiableFunction<T>* f : *fs){
			uint32_t nparams = f->get_number_parameters();
			f->align_parameters(p+offset);
			offset += nparams;
		}
	}
	virtual void align_parameters(std::vector<T*> p){
		assert(p.size() == get_number_parameters());
		std::vector<DifferentiableFunction<T>*>* fs = get_sub_functions();
		uint32_t offset = 0;
		for(DifferentiableFunction<T>* f : *fs){
			uint32_t nparams = f->get_number_parameters();
			f->align_parameters(std::vector<T*>(p.begin()+offset, p.begin()+offset+nparams));
			offset += nparams;
		}
	}
};

template<typename T>
class Sum: public virtual CombinationOfElementalFunctions<T> {
	std::vector<DifferentiableFunction<T>*> functions;
public:
	Sum(std::initializer_list<DifferentiableFunction<T>*> funs){
		functions.assign(funs.begin(), funs.end());
	}
	Sum(std::vector<DifferentiableFunction<T>*> funs){
		functions.assign(funs.begin(), funs.end());
	}
	virtual std::vector<DifferentiableFunction<T>*>* const get_sub_functions() {
		return &functions;
	}
	virtual T const evaluate(T x){
		T res = 0.0;
		for(DifferentiableFunction<T>* f : functions){
			res += f->evaluate(x);
		}
		return res;
	}
	virtual T const derivative(const T x){
		T res = 0.0;
		for(DifferentiableFunction<T>* f : functions){
			res += f->derivative(x);
		}
		return res;
	}
	virtual T const derivative(const T x, T* p){
		if(!this->depends_directly_on(p)){ 
			return 0.0;
		}
		T res = 1.0;
		for(DifferentiableFunction<T>* f : functions){
			res += f->derivative(x, p);
		}
		return res;
	}
};

template<typename T>
class Product: public virtual CombinationOfElementalFunctions<T> {
	std::vector<DifferentiableFunction<T>*> functions;
public:
	Product(std::initializer_list<DifferentiableFunction<T>*> funs){
		functions.assign(funs.begin(), funs.end());
	}
	Product(std::vector<DifferentiableFunction<T>*> funs){
		functions.assign(funs.begin(), funs.end());
	}
	virtual std::vector<DifferentiableFunction<T>*>* const get_sub_functions() {
		return &functions;
	}
	virtual T const evaluate(T x){
		T res = 1.0;
		for(DifferentiableFunction<T>* f : functions){
			res *= f->evaluate(x);
		}
		return res;
	}
	virtual T const derivative(const T x){
		T res = 0.0;
		for(uint32_t ifun=0; ifun<functions.size(); ifun ++){
			T part = 1.0;
			for(uint32_t jfun=0; jfun<functions.size(); jfun++){
				if(ifun == jfun){
					part *= functions[jfun]->derivative(x);
				}
				else{
					part *= functions[jfun]->evaluate(x);
				}
			}
			res += part;
		}
		return res;
	}
	virtual T const derivative(const T x, T* p){
		if(!this->depends_directly_on(p)){ 
			return 0.0;
		}
		T res = 0.0;
		for(uint32_t ifun=0; ifun<functions.size(); ifun ++){
			T part = 1.0;
			for(uint32_t jfun=0; jfun<functions.size(); jfun++){
				if(ifun == jfun){
					part *= functions[jfun]->derivative(x, p);
				}
				else{
					part *= functions[jfun]->evaluate(x);
				}
			}
			res += part;
		}
		return res;
	}
};

template<typename T>
class CompositeFunction: public virtual CombinationOfElementalFunctions<T> {
	std::vector<DifferentiableFunction<T>*> functions;
	std::vector<T> get_sub_evaluations(const T x){
		std::vector<T> evals(functions.size());
		T y = x;
		uint32_t iFun = 0;
		std::generate(evals.begin(), evals.end(), [&y, &iFun, this](){
				T ret = y;
				y = this->functions[iFun++]->evaluate(y);
				return ret;
			});
		return evals;
	}
public:
	CompositeFunction(std::vector<DifferentiableFunction<T>*> funs){
		functions.assign(funs.begin(), funs.end());
	}
	virtual std::vector<DifferentiableFunction<T>*>* const get_sub_functions() {
		return &functions;
	}
	virtual T const evaluate(T x){
		T res = x;
		for(DifferentiableFunction<T>* f : functions){
			res = f->evaluate(res);
		}
		return res;
	}
	virtual T const derivative(const T x){
		T res = 1.0;
		std::vector<T> evals = get_sub_evaluations(x);
		for(uint32_t iFunction = functions.size() - 1; iFunction >= 0; iFunction--){
			res *= functions[iFunction]->derivative(evals[iFunction]);
		}
		return res;
	}
	virtual T const derivative(const T x, T* p){
		if(!this->depends_directly_on(p)){ 
			return 0.0;
		}
		T res = 1.0;
		std::vector<T> evals = get_sub_evaluations(x);
		uint32_t iFunction = functions.size() - 1;
		while(!functions[iFunction]->depends_directly_on(p)){
			res *= functions[iFunction]->derivative(evals[iFunction]);
			iFunction--;
			assert(iFunction >= 0);
		}
		res *= functions[iFunction]->derivative(evals[iFunction],p);
		return res;
	}
};

template<typename T>
class LinearFunctionComposer {
	std::vector<DifferentiableFunction<T>*>* functions;
public:
	LinearFunctionComposer(){
		functions = new std::vector<DifferentiableFunction<T>*>();
	}
	~LinearFunctionComposer(){
		if(functions) delete functions;
	}
	template<class c> DifferentiableFunction<T>* apply_after(){
		static_assert(c::is_elemental, "Expecting c to be derived from ElementalFunction");
		functions->emplace_back(c::get_instance());
		return functions->back();
	}
	void apply_after(DifferentiableFunction<T>* f){
		functions->push_back(f);
	}
	CompositeFunction<T> done(){
		CompositeFunction<T> c = CompositeFunction<T>(*functions);
		functions = nullptr;
		return c;
	}
};
