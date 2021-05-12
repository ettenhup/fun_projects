#include <cstdint>
#include <vector>
#include "DifferentiableFunction.h"
#pragma once

template<typename T>
class Node;

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

	std::vector<Edge<T> >* get_children(){
		return &children;
	}

	std::vector<Edge<T> >* get_parents(){
		return &parents;
	}

	uint32_t get_number_parents(){
		return parents.size;
	}

	uint32_t get_number_children(){
		return children.size;
	}
};
