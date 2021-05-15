template<typename T>
class Datatype{
};
template<typename T>
class Parameter: public Datatype<T>{
	T* p;
public:
	Parameter(){};
	Parameter(T* p):p(p){};
};
template<typename T>
class Constant: public Datatype<T>{
	T p;
	Constant(){};
public:
	Constant(T* p):p(*p){};
	Constant(T p):p(p){};
};
template<typename T>
class Variable: public Datatype<T>{
	T p;
public:
	Variable(T p):p(p){};
};
