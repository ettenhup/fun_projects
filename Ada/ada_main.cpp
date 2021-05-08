#include <cstdint>
#include <initializer_list>
#include <cstdarg>
#include <math.h>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <stdexcept>
#include <random>
#include <functional>
//#include <Eigen/Dense>
 
//using Eigen::MatrixXd;
 
// clang++ -I/usr/local/Cellar/eigen/3.3.8_1/include/eigen3 ada_main.cpp -o ada

template<typename T>
class DifferentiableFunction{
public:
	enum EnumDiff { is_differentiable_function = true };
	static constexpr uint32_t get_number_parameters(){
		return 0;
	};
	static DifferentiableFunction<T>* get_instance(T* p){
		return nullptr;
	}
	virtual void get_parameters(const T** p) = 0;
	virtual T evaluate(const T x) = 0;
	virtual T derivative(const T x) = 0;
	virtual T derivative(const T x, T* p) = 0;
};

template<typename T>
class ReLu: public DifferentiableFunction<T> {
public:
	static constexpr uint32_t get_number_parameters(){
		return 0;
	}
	static DifferentiableFunction<T>* get_instance(T* p){
		return new ReLu<T>();
	}
	virtual void get_parameters(const T** p){
	}
	virtual T evaluate(const T x){
		return x > 0 ? x : static_cast<T>(0);
	}
	virtual T derivative(const T x){
		return x > 0 ?static_cast<T>(1) : static_cast<T>(0);
	}
	virtual T derivative(const T x, T* p){
		return static_cast<T>(0);
	}
};

template<typename T>
class Exp: public DifferentiableFunction<T> {
public:
	static constexpr uint32_t get_number_parameters(){
		return 0;
	}
	static DifferentiableFunction<T>* get_instance(T* p){
		return new Exp<T>();
	}
	virtual void get_parameters(const T** p){
	}
	virtual T evaluate(const T x){
		return exp(x);
	}
	virtual T derivative(const T x){
		return exp(x);
	}
	virtual T derivative(const T x, T* p){
		return static_cast<T>(0);
	}
};

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-1.0,1.0);

template<typename T>
class GaussCurve: public DifferentiableFunction<T>{
	const T *a, *b, *c;
public:
	GaussCurve(const T* a, const T* b, const T* c) : a(a), b(b), c(c){};
	GaussCurve(const T* a) : a(a), b(a+1), c(a+2){};
	static DifferentiableFunction<T>* get_instance(T* p){
		return new GaussCurve<T>(p);
	}
	static constexpr uint32_t get_number_parameters(){
		return 3;
	}
	virtual void get_parameters(const T** p){
		p[0] = a;
		p[1] = b;
		p[2] = c;
	}
	virtual T evaluate(const T x){
		const T d = x - (*c);
		return (*a) * exp((*b) * d * d); 
	}
	virtual T derivative(const T x){
		return 2*(*b)*(x-(*c))*evaluate(x);
	}
	virtual T derivative(const T x,T* p){
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
};

template<typename T>
class Edge;

template<typename T>
class Node{
	std::vector<Edge<T> > parents;
	std::vector<Edge<T> > children;
	DifferentiableFunction<T>* activation;
	Node(){};
public:
	Node(DifferentiableFunction<T>* f){
		activation = f;
	}

	void add_parent(Node<T>* p, DifferentiableFunction<T>* f){
		parents.emplace_back(p, this, f);
	}
	void add_child(Node<T>* c, DifferentiableFunction<T>* f){
		children.emplace_back(this, c, f);
	}

	void set_parents(std::vector<Node<T>*> p, std::vector<DifferentiableFunction<T>*> f){
		assert(p.size() == f.size());
		for(uint32_t i = 0; i< p.size(); i++){
			add_parent(p[i], f[i]);
			p[i]->add_child(this, f[i]);
		}
	}

	void set_children(std::vector<Node<T>*> c, std::vector<DifferentiableFunction<T>*> f){
		assert(c.size() == f.size());
		for(uint32_t i = 0; i< c.size(); i++){
			add_child(c[i], f[i]);
			c[i]->add_parent(this, f[i]);
		}
	}

	DifferentiableFunction<T> * get_activation(){
		return activation;
	}

	uint32_t get_number_parents(){
		return parents.size;
	}

	uint32_t get_number_children(){
		return children.size;
	}
};

template<typename T>
class Edge{
	Node<T>* from;
	Node<T>* to;
	DifferentiableFunction<T>* f;
	Edge(){};
public:
	Edge(Node<T>* from, Node<T>* to, DifferentiableFunction<T>* f){
		this->from = from;
		this->to = to;
		this->f = f;
	}

	DifferentiableFunction<T> * get_function(){
		return f;
	}
};

template<typename T>
class CompositeFunction: public DifferentiableFunction<T> {
	// This should have a memory optimized parameters layout and not
	std::vector<std::vector<T> > parameters;
	std::vector<Node<T> > nodes;
	std::vector<DifferentiableFunction<T> > functions;

public:
	CompositeFunction(std::vector<Node<T> > n, std::vector<T> parameters){
		parameters.push_back(parameters);
		nodes = n;
	}
};

template<typename T>
class FunctionComposer {
	std::vector<Node<T> > allNodes;
	std::vector<DifferentiableFunction<T>*> allFunctions;
	std::vector<T> parameters;
public:
	FunctionComposer(){
	}

	T* reserve_parameters(std::initializer_list<T> args){
		for (auto e : args) {
			parameters.push_back(e);
		}
		return &(*(parameters.end() - args.size()));
	}

	Node<T>* add_node(DifferentiableFunction<T>* f){
	}

	void add_edge(Node<T>* from, Node<T>* to, DifferentiableFunction<T>* f){
		from->add_child(to, f);
		to->add_parent(from, f);
	}

	CompositeFunction<T> done(){
		return CompositeFunction<T>(allNodes, parameters);
	}

	template<class c> Node<T>* add_node(){
		static_assert(c::is_differentiable_function, "Expecting c to be derived from DifferentiableFunction");
		static_assert(c::get_number_parameters == 0, "Expecting c to be have zero parameters");
		allFunctions.emplace_back(c::get_instance(nullptr));
		allNodes.emplace_back(allFunctions.back());
		return &allNodes.back();
	}

	template<class c> Node<T>* add_node(std::initializer_list<T> args){
		static_assert(c::is_differentiable_function, "Expecting c to be derived from DifferentiableFunction");
		//static_assert(c::get_number_parameters() == args.size(), "Expecting c to be have the provided number of parameters");
		double* p = reserve_parameters(args);
		allFunctions.emplace_back(c::get_instance(p));
		allNodes.emplace_back(allFunctions.back());
		return &allNodes.back();
	}
};


int main()
{
	double a=0.3;
	double b=0.4;
	double c=0.5;
	GaussCurve<double> gc(&a,&b,&c);
	std::cout << gc.evaluate(1) << " "<< gc.evaluate(2) << std::endl;
	std::cout << gc.derivative(2,&a) << " "<< gc.derivative(2,&b) << " " << gc.derivative(2,&c) << std::endl;

	FunctionComposer<double> fc;
	double* p = fc.reserve_parameters({0.3, 0.4, 0.5});
	std::cout << *p << " " << *(p+1) << " " << *(p+2) << std::endl;
	GaussCurve<double> test(p);
	std::cout << test.evaluate(1) << " "<< test.evaluate(2) << std::endl;
	std::cout << test.derivative(2,p) << " "<< test.derivative(2,p+1) << " " << test.derivative(2,p+2) << std::endl;
	Node<double>* gn = fc.add_node<GaussCurve<double>>({0.3, 0.4, 0.5});
	std::cout << gn->get_activation()->evaluate(1) << " "<< gn->get_activation()->evaluate(2) << std::endl;
	double* param()
	std::cout << gn->get_activation()->derivative(2,p) << " "<< gn->get_activation()->derivative(2,p+1) << " " << gn->get_activation()->derivative(2,p+2) << std::endl;


  // MatrixXd m(2,2);
  // m(0,0) = 3;
  // m(1,0) = 2.5;
  // m(0,1) = -1;
  // m(1,1) = m(1,0) + m(0,1);
  // std::cout << m << std::endl;
}
