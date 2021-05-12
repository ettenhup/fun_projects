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
//	LinearCompositeFunction<T> done(){
//		return LinearCompositeFunction<T>(allNodes, parameters);
//	}
//};


int main()
{
	double a=0.3;
	double b=0.4;
	double c=0.5;
	GaussCurve<double> gc(&a,&b,&c);
	std::cout << gc.evaluate(1) << " "<< gc.evaluate(2) << std::endl;
	std::cout << gc.derivative(2,&a) << " "<< gc.derivative(2,&b) << " " << gc.derivative(2,&c) << std::endl;

	LinearFunctionComposer<double> fc;
	fc.apply_after<Linear<double>>({2.0, 2.0});
	fc.apply_after<Linear<double>>({4.0, 0.5});
	//// LinearCompositeFunction<double>* composite = nullptr;
	//// std::vector<double>* parameters = nullptr;
	auto composite  = fc.done();
	//assert(composite!=nullptr);
	std::cout << " c: " << composite.first.evaluate(1) << " "<< composite.first.evaluate(2) << std::endl;

	double m=2.0;
	Linear<double> linear(&m,&c);
	std::cout << linear.evaluate(1) << " "<< linear.evaluate(2) << std::endl;

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
