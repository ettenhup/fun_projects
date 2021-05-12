#include <cstdint>
#include <initializer_list>
#include <cstdarg>
#include <vector>
#include <random>
#include <iostream>
#include "DifferentiableFunction.h"
#pragma once
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0,1.0);

template<typename T>
class LinearCompositeFunction: public DifferentiableFunction<T> {
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
	LinearCompositeFunction(std::vector<DifferentiableFunction<T>*> funs){
		functions.assign(funs.begin(), funs.end());
	}
	virtual uint32_t const get_number_parameters(){
		return std::accumulate(functions.begin(), functions.end(), 0, [](uint32_t acc, DifferentiableFunction<T>* f){
				return acc + f->get_number_parameters();
			});
	}
	virtual void const get_parameters(T** p){
		uint32_t totParam = 0;
		for(DifferentiableFunction<T>* f : functions){
			uint32_t nparam = f->get_number_parameters();
			if(nparam>0){
				T** funParams = new T*[nparam];
				f->get_parameters(funParams);
				for(uint32_t iparam = 0; iparam < nparam; iparam++){
					p[totParam++] = funParams[iparam];
				}
				delete [] funParams;
			}
		}
	}
	virtual T const evaluate(T x){
		T res = x;
		std::cout<<std::endl;
		for(DifferentiableFunction<T>* f : functions){
			std::cout << " (" << res  << ", " << f->evaluate(1) << ") ";
			res = f->evaluate(res);
		}
		std::cout << " " << res << std::endl;
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
		if(!depends_directly_on(p)){ 
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
	virtual bool const depends_directly_on(T* p) {
		return std::any_of(functions.begin(), functions.end(), [&p](DifferentiableFunction<T>* f){return f->depends_directly_on(p);});
	}
};
template<typename T>
class LinearFunctionComposer {
	std::vector<DifferentiableFunction<T>*>* functions;
	std::vector<T>* parameters;

	T* reserve_parameters(std::initializer_list<T> args){
		for (auto e : args) {
			parameters->push_back(e);
		}
		return &(*(parameters->end() - args.size()));
	}
	T* reserve_parameters(uint32_t n){
		for(uint32_t iparam = 0; iparam < n; iparam++){
			parameters->push_back(dis(rd));
		}
		return &(*(parameters->end() - n));
	}

public:
	LinearFunctionComposer(){
		functions = new std::vector<DifferentiableFunction<T>*>();
		parameters = new std::vector<T>();
	}

	template<class c> DifferentiableFunction<T>* apply_after(){
		static_assert(c::is_elemental, "Expecting c to be derived from DifferentiableFunction");
		T* p =  c::get_static_number_parameters == 0 ? nullptr : reserve_parameters(c::get_static_number_parameters);
		functions->emplace_back(c::get_instance(p));
		return functions->back();
	}

	template<class c> DifferentiableFunction<T>* apply_after(std::initializer_list<T> args){
		static_assert(c::is_elemental, "Expecting c to be derived from DifferentiableFunction");
		//assert(c::get_static_number_parameters() == args.size());
		T* p = reserve_parameters(args);
		functions->emplace_back(c::get_instance(p));
		return functions->back();
	}

	std::pair<LinearCompositeFunction<T>, std::vector<T> > done(){
		return std::make_pair(LinearCompositeFunction<T>(*functions), *parameters);
	}
};
