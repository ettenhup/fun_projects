#include <cstdint>
#include <cstdarg>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>
#include <stdexcept>
#include <random>
#include <functional>
#include "assert.h"
#include "DifferentiableFunction.h"
#include "ElementalFunctions.h"
#include "Node.h"
#include "EdgeFunction.h"
//#include <Eigen/Dense>
//using Eigen::MatrixXd;
// clang++ -I/usr/local/Cellar/eigen/3.3.8_1/include/eigen3 ada_main.cpp -o ada


//template<typename T>
//class FunctionComposer {
//	std::vector<Node<T>*> allNodes;
//	std::vector<DifferentiableFunction<T>*> allFunctions;
//	std::vector<T> parameters;
//
//	T* reserve_parameters(std::initializer_list<T> args){
//		for (auto e : args) {
//			parameters.push_back(e);
//		}
//		return &(*(parameters.end() - args.size()));
//	}
//
//public:
//	FunctionComposer(){
//	}
//
//	template<class c> Edge<T>* add_edge(Node<T>* from, Node<T>* to, std::initializer_list<T> args){
//		static_assert(c::is_differentiable_function, "Expecting c to be derived from DifferentiableFunction");
//		//static_assert(c::get_static_number_parameters() == args.size(), "Expecting c to be have the provided number of parameters");
//		double* p = reserve_parameters(args);
//		allFunctions.emplace_back(c::get_instance(p));
//		from->add_child(to, allFunctions.back());
//		to->add_parent(from, allFunctions.back());
//		return &from->get_children()->back();
//	}
//
//	template<class c> Node<T>* add_node(){
//		static_assert(c::is_differentiable_function, "Expecting c to be derived from DifferentiableFunction");
//		static_assert(c::get_static_number_parameters == 0, "Expecting c to be have zero parameters");
//		allFunctions.emplace_back(c::get_instance(nullptr));
//		allNodes.emplace_back(allFunctions.back());
//		return &allNodes.back();
//	}
//
//	template<class c> Node<T>* add_node(std::initializer_list<T> args){
//		static_assert(c::is_differentiable_function, "Expecting c to be derived from DifferentiableFunction");
//		//static_assert(c::get_static_number_parameters() == args.size(), "Expecting c to be have the provided number of parameters");
//		double* p = reserve_parameters(args);
//		allFunctions.emplace_back(c::get_instance(p));
//		allNodes.emplace_back(allFunctions.back());
//		return &allNodes.back();
//	}
//
//	CompositeFunction<T> done(){
//		return CompositeFunction<T>(allNodes, parameters);
//	}
//};

template<typename T>
class SimpleFunctionOptimizer{
	virtual void fit(std::vector<T>* x, std::vector<T>* y) = 0; 
};

template<typename T>
class GradientDescentOptimizer: public virtual SimpleFunctionOptimizer<T> {
	DifferentiableFunction<T>* f;
	T learning_rate;
public:
	GradientDescentOptimizer(DifferentiableFunction<T>* f){
		this->f = f;
		learning_rate = 0.001;
	}

	void set_learning_rate(T learning_rate){
		this->learning_rate = learning_rate;
	}

	virtual void fit(std::vector<T>* x, std::vector<T>* y){
		assert(x->size() == y->size());
		uint32_t nParam = f->get_number_parameters();
		std::vector<T*> params = f->get_parameters();

		int niter = 0;
		bool converged = false;
		while(niter++ < 100 && !converged){
			std::vector<T> gradient(0.0, nParam);

			for(uint32_t iSample = 0; iSample < nParam; iSample++){
				for(uint32_t iParam=0; iParam<nParam; iParam++){
					gradient[iParam] += f->derivative(x->at(iSample), params[iParam]);
				}
			}

			std::cout << "Iteration " << niter << " Gradient sum" << accumulate(gradient.begin(),gradient.end(),0) << std::endl;

			for(uint32_t iParam=0; iParam<nParam; iParam++){
				*params[iParam] = params[iParam] - learning_rate * gradient[iParam];
			}
		}
	}
};

int main()
{
	double params[] = {0.3, 0.4, 0.5};
	GaussCurve<double> gc1;
	gc1.align_parameters(&params[0]);
	std::cout << gc1.evaluate(1) << " "<< gc1.evaluate(2) << std::endl;
	std::cout << gc1.derivative(2,&params[0]) << " "<< gc1.derivative(2,&params[1]) << " " << gc1.derivative(2,&params[2]) << std::endl;
	double d=1.3;
	double e=1.4;
	double f=1.5;
	GaussCurve<double> gc2;
	gc2.align_parameters(std::vector<double*>{&d, &e, &f});
	std::cout << gc2.evaluate(1) << " "<< gc2.evaluate(2) << std::endl;
	std::cout << gc2.derivative(2,&d) << " "<< gc2.derivative(2,&e) << " " << gc2.derivative(2,&f) << std::endl;

	Product<double> prodOfGauss({&gc1, &gc2});
	Sum<double> sumOfGauss({&gc1, &gc2});
	std::cout << "prod: " << prodOfGauss.evaluate(1) << " "<< prodOfGauss.evaluate(2) << std::endl;
	std::cout << "sum: "<< sumOfGauss.evaluate(1) << " "<< sumOfGauss.evaluate(2) << std::endl;

	LinearFunctionComposer<double> fc1;
	fc1.apply_after<Linear<double>>();
	fc1.apply_after<GaussCurve<double>>();
	CompositeFunction<double> fc1c = fc1.done();
	std::vector<double> fc1c_params {1.0, 0.0, 0.3, 0.4, 0.5};
	fc1c.align_parameters(&fc1c_params[0]);

	LinearFunctionComposer<double> fc;
	fc.apply_after<Linear<double>>();
	fc.apply_after<Linear<double>>();
	fc.apply_after(&fc1c);

	CompositeFunction<double> composite  = fc.done();
	std::vector<double> fc_params {5.0, -6.0, -4.0, -3.0,1.0, 0.0, 0.3, 0.4, 0.5};
	composite.align_parameters(&fc_params[0]);
	std::cout << " c: " << composite.evaluate(1) << " "<< composite.evaluate(19.0/20.0) << std::endl;
	std::cout << " n: " << composite.get_number_parameters() << std::endl;
	std::vector<double*> check_p = composite.get_parameters();
	std::cout << " p: ";
	for(const auto& prm : check_p){
		std::cout << *prm <<" ";
	}
	std::cout << std::endl;

	//Node<double>* gn = fc.add_node<GaussCurve<double>>({0.3, 0.4, 0.5});
	//std::cout << gn->get_activation()->evaluate(1) << " "<< gn->get_activation()->evaluate(2) << std::endl;
	////std::cout << gn->get_activation()->derivative(2,p) << " "<< gn->get_activation()->derivative(2,p+1) << " " << gn->get_activation()->derivative(2,p+2) << std::endl;
	//Node<double>* lin = fc.add_node<Linear<double>>({1.0, 2.0});
	//std::cout << lin->get_activation()->evaluate(1) << " "<< lin->get_activation()->evaluate(2) << std::endl;
	//Edge<double>* e = fc.add_edge<Linear<double>>(lin, gn,{1,0});


  // MatrixXd m(2,2);
  // m(0,0) = 3;
  // m(1,0) = 2.5;
  // m(0,1) = -1;
  // m(1,1) = m(1,0) + m(0,1);
  // std::cout << m << std::endl;
}
