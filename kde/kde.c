////////////////////////////////////////////////////////////
/////         KDE CLASS - C IMPLEMENTATION            //////
///// copyright (C) 2014 Martin Leuermann (May 2014)  //////
////////////////////////////////////////////////////////////
//
// MIT License
//
// Copyright (c) 2014-2019 Martin Leuermann
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <Python.h>
#include <math.h>
#include <time.h>

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

int diff_ms(struct timeval t1, struct timeval t2)
{
	return (((t1.tv_sec - t2.tv_sec) * 1000000) + (t1.tv_usec - t2.tv_usec))/1000;
}

///////////////////////////////////////////////////
//////// CALCULATE KDE VALUES FOR /////////////////
//////// DATASET X AND POINTS Y   /////////////////
///////////////////////////////////////////////////

////////////////////////
/// FOR 1 DIMENSION ///
////////////////////////
static PyObject *pr_kde_1d(PyObject *self, PyObject *args){

	////////////////// DECLARATIONS ///////////////////////
	int ListSize1, ListSize2, i,j;
	double c, detC, res, ent, d, h;
	PyObject *objx, *objy, *objpreFac, *objw_norm;
	PyObject *ListItem, *ListItem2, *ListItem3;
	double *x, *y, *preFac, *w_norm;

	/////////////////// GET INPUT //////////////////////////
	if (!PyArg_ParseTuple(args, "dOOdOO", &c, &objx, &objy, &h, &objpreFac, &objw_norm))
		return NULL;

	d = 1.0;
	detC = c;

	////////// GET FIRST LIST-GROUP FROM PYTHON ////////////
	ListSize1 = PyList_Size(objx);
	x = (double*) malloc(sizeof(double)*ListSize1);
	preFac = (double*) malloc(sizeof(double)*ListSize1);
	w_norm = (double*) malloc(sizeof(double)*ListSize1);

	for(i=0; i < ListSize1; i++ ) {
		ListItem = PyList_GetItem(objx, i);
		ListItem2 = PyList_GetItem(objw_norm, i);
		ListItem3 = PyList_GetItem(objpreFac, i);
		if( PyFloat_Check(ListItem) && PyFloat_Check(ListItem2) && PyFloat_Check(ListItem3) ){
			x[i] = PyFloat_AsDouble(ListItem);
			w_norm[i] = PyFloat_AsDouble(ListItem2);
			preFac[i] = PyFloat_AsDouble(ListItem3);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	////////// GET SECOND LIST-GROUP FROM PYTHON ////////////
	ListSize2 = PyList_Size(objy);
	y = (double*) malloc(sizeof(double)*ListSize2);

	for(i=0; i < ListSize2; i++ ) {
		ListItem = PyList_GetItem(objy, i);
		if( PyFloat_Check(ListItem) ) {
			y[i] = PyFloat_AsDouble(ListItem);
		}else{
			printf("Error: lists contain a non-float value.\n");
		 	exit(1);
		}
	}

	/////////////// RUN CALCULATIONS /////////////////////
	PyObject *pylist;
	pylist = PyList_New(ListSize2);

	for(i=0; i < ListSize2; i++) {
		res = 0.0;
		for (j=0; j < ListSize1; j++) {
			ent = x[j]-y[i];
			res += w_norm[j] * exp(preFac[j] * c*pow(ent, 2));
		}
		PyList_SET_ITEM(pylist, i, PyFloat_FromDouble(res));
	}

	///////// FREE MEMORY ///////////
	free(x);
	free(preFac);
	free(w_norm);

	///////// RETURN RESULTING VALUES FOR y /////////
	return pylist;
}


////////////////////////
/// FOR 2 DIMENSIONS ///
////////////////////////
static PyObject *pr_kde_2d(PyObject *self, PyObject *args){

	////////////////// DECLARATIONS ///////////////////////
	int ListSize1, ListSize2, i,j;
	double c11, c12, c21, c22, detC, res, ent1, ent2, d, h;
	PyObject *objx1, *objx2, *objy1, *objy2, *objpreFac, *objw_norm;
	PyObject *ListItem, *ListItem2, *ListItem3, *ListItem4;
	double *x1, *x2, *y1,*y2, *preFac, *w_norm;

	/////////////////// GET INPUT //////////////////////////
	if (!PyArg_ParseTuple(args, "ddddOOOOdOO", &c11, &c12, &c21, &c22, &objx1, &objx2, &objy1, &objy2, &h, &objpreFac, &objw_norm))
		return NULL;

	d = 2.0;
	detC = c11*c22 - c12*c21;

	////////// GET FIRST LIST-GROUP FROM PYTHON ////////////
	ListSize1 = PyList_Size(objx2);
	x1 = (double*) malloc(sizeof(double)*ListSize1);
	x2 = (double*) malloc(sizeof(double)*ListSize1);
	preFac = (double*) malloc(sizeof(double)*ListSize1);
	w_norm = (double*) malloc(sizeof(double)*ListSize1);

	for(i=0; i < ListSize1; i++ ) {
		ListItem = PyList_GetItem(objx1, i);
		ListItem2 = PyList_GetItem(objx2, i);
		ListItem3 = PyList_GetItem(objpreFac, i);
		ListItem4 = PyList_GetItem(objw_norm, i);
		if( PyFloat_Check(ListItem) && PyFloat_Check(ListItem2) && PyFloat_Check(ListItem3) && PyFloat_Check(ListItem4) ){
			x1[i] = PyFloat_AsDouble(ListItem);
			x2[i] = PyFloat_AsDouble(ListItem2);
			preFac[i] = PyFloat_AsDouble(ListItem3);
			w_norm[i] = PyFloat_AsDouble(ListItem4);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	////////// GET SECOND LIST-GROUP FROM PYTHON ////////////
	ListSize2 = PyList_Size(objy1);
	y1 = (double*) malloc(sizeof(double)*ListSize2);
	y2 = (double*) malloc(sizeof(double)*ListSize2);

	for(i=0; i < ListSize2; i++ ) {
		ListItem = PyList_GetItem(objy1, i);
		ListItem2 = PyList_GetItem(objy2, i);
		if( PyFloat_Check(ListItem) && PyFloat_Check(ListItem2) ) {
			y1[i] = PyFloat_AsDouble(ListItem);
			y2[i] = PyFloat_AsDouble(ListItem2);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	/////////////// RUN CALCULATIONS /////////////////////
	PyObject *pylist;

	pylist = PyList_New(ListSize2);

	for(i=0; i < ListSize2; i++) {
		res = 0.0;
		for (j=0; j < ListSize1; j++) {
			ent1 = x1[j]-y1[i];
			ent2 = x2[j]-y2[i];
			res += w_norm[j] * exp(preFac[j] * (ent1*(c11*ent1+c12*ent2) + ent2*(c21*ent1+c22*ent2)));
		}
		PyList_SET_ITEM(pylist, i, PyFloat_FromDouble(res));
	}

	///////// FREE MEMORY ///////////
	free(x1);
	free(x2);
	free(preFac);
	free(w_norm);

	///////// RETURN RESULTING VALUES FOR y /////////
	return pylist;
}



///////////////////////////////////////////////////
//////// GET LAMBDA FOR DATASET OF X //////////////
///////////////////////////////////////////////////

////////////////////////
/// FOR 1 DIMENSION ///
////////////////////////

static PyObject *pr_getLambda_1d(PyObject *self, PyObject *args){

	////////////////// DECLARATIONS ///////////////////////
	int ListSize1, i,j, d;
	double c, detC, thisKde, ent, invGlob, logSum, alpha, h, preFac; //, tempNorm, weight, tempNormOld;
	PyObject *objx;
	PyObject *ListItem, *ListItem2;
	PyObject *obj_w_norm;
	double *x, *lambda, *kde, *w_norm;

	/////////////////// GET INPUT //////////////////////////
	if (!PyArg_ParseTuple(args, "dOOdd", &c, &objx, &obj_w_norm, &h, &alpha ) )
		return NULL;

	////////// GET FIRST LIST-GROUP FROM PYTHON ////////////
	ListSize1 = PyList_Size(objx);
	x = (double*) malloc(sizeof(double)*ListSize1);
	lambda = (double*) malloc(sizeof(double)*ListSize1);
	kde = (double*) malloc(sizeof(double)*ListSize1);
	w_norm = (double*) malloc(sizeof(double)*ListSize1);

	for(i=0; i < ListSize1; i++ ) {
		ListItem = PyList_GetItem(objx, i);
		ListItem2 = PyList_GetItem(obj_w_norm, i);

		if( PyFloat_Check(ListItem) && PyFloat_Check(ListItem2) ) {
			x[i] = PyFloat_AsDouble(ListItem);
			w_norm[i] = PyFloat_AsDouble(ListItem2);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	/////////////// RUN CALCULATIONS /////////////////////
	PyObject *lambdaList;
	lambdaList = PyList_New(ListSize1);

	invGlob = 0.0;
	logSum = 0.0;
	d = 1.0;
	detC = c;
	preFac = -0.5/pow(h, 2);

	for(i=0; i < ListSize1; i++) {
		thisKde = 0.0;
		for (j=0; j < ListSize1; j++) {
			ent = x[j]-x[i];
			thisKde += w_norm[j] * exp(preFac * (ent*c*ent));
		}
		logSum += 1.0/ListSize1 * log(thisKde);
		kde[i] = thisKde;
	}
	invGlob = 1.0/exp(logSum);

	for(i=0; i< ListSize1; i++) {
		lambda[i] = 1.0/pow(invGlob*kde[i], alpha);
		PyList_SET_ITEM(lambdaList, i, PyFloat_FromDouble(lambda[i]));
	}

	///////// FREE MEMORY ///////////
	free(x);
	free(lambda);
	free(kde);
	free(w_norm);

	///////// RETURN RESULTING VALUES FOR LAMBDAS /////////
	return lambdaList;
}

////////////////////////
/// FOR 2 DIMENSIONS ///
////////////////////////

static PyObject *pr_getLambda_2d(PyObject *self, PyObject *args){
	////////////////// DECLARATIONS ///////////////////////
	int ListSize1, i,j, d;
	double c11, c12, c21, c22, detC, thisKde, ent1, ent2, invGlob, logSum, alpha, h, preFac; // , tempNorm, weight;
	PyObject *objx1, *objx2, *obj_w_norm;
	PyObject *ListItem, *ListItem2, *ListItem3;
	double *x1, *x2, *lambda, *kde, *w_norm;

	/////////////////// GET INPUT //////////////////////////
	if (!PyArg_ParseTuple(args, "ddddOOOdd", &c11, &c12, &c21, &c22, &objx1, &objx2, &obj_w_norm, &h, &alpha ) )
		return NULL;

	////////// GET FIRST LIST-GROUP FROM PYTHON ////////////
	ListSize1 = PyList_Size(objx2);
	x1 = (double*) malloc(sizeof(double)*ListSize1);
	x2 = (double*) malloc(sizeof(double)*ListSize1);
	lambda = (double*) malloc(sizeof(double)*ListSize1);
	kde = (double*) malloc(sizeof(double)*ListSize1);
	w_norm = (double*) malloc(sizeof(double)*ListSize1);

	for(i=0; i < ListSize1; i++ ) {
		ListItem = PyList_GetItem(objx1, i);
		ListItem2 = PyList_GetItem(objx2, i);
		ListItem3 = PyList_GetItem(obj_w_norm, i);

		if( PyFloat_Check(ListItem) && PyFloat_Check(ListItem2) && PyFloat_Check(ListItem3)) {
			x1[i] = PyFloat_AsDouble(ListItem);
			x2[i] = PyFloat_AsDouble(ListItem2);
			w_norm[i] = PyFloat_AsDouble(ListItem3);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	/////////////// RUN CALCULATIONS /////////////////////
	PyObject *lambdaList;
	lambdaList = PyList_New(ListSize1);


	invGlob = 0.0;
	logSum = 0.0;
	d = 2.0;
	detC = c11*c22 - c12*c21;
	preFac = -0.5/(h*h);

	for(i=0; i < ListSize1; i++) {
		thisKde = 0.0;
		for (j=0; j < ListSize1; j++) {
			ent1 = x1[j]-x1[i];
			ent2 = x2[j]-x2[i];
			thisKde += w_norm[j] * exp(preFac * (ent1*(c11*ent1+c12*ent2) + ent2*(c21*ent1+c22*ent2)));
		}
		logSum += 1.0/ListSize1 * log(thisKde);
		kde[i] = thisKde;
	}
	invGlob = 1.0/exp(logSum);

	for(i=0; i< ListSize1; i++) {
		lambda[i] = 1.0/pow(invGlob*kde[i], alpha);
		PyList_SET_ITEM(lambdaList, i, PyFloat_FromDouble(lambda[i]));
	}

	///////// FREE MEMORY ///////////
	free(x1);
	free(x2);
	free(lambda);
	free(kde);
	free(w_norm);

	///////// RETURN RESULTING VALUES FOR LAMBDAS /////////
	return lambdaList;
}

////////////////////////
/// FOR N DIMENSIONS ///
////////////////////////

static PyObject *pr_kde_ND(PyObject *self, PyObject *args){

	////////////////// DECLARATIONS ///////////////////////
	int ndata, neval, nelems_c, ndim, i,j,d,k;
	double det_inv_cov, h;
	PyObject *obj_entries, *obj_preFac, *obj_w_norm, *obj_evals;
	PyObject *ListItem, *ListItem2, *obj_inv_cov;
	double *x, *preFac, *w_norm, *inv_cov, *evals;

	/////////////////// GET INPUT //////////////////////////
	if (!PyArg_ParseTuple(args, "iOOOOOdd", &ndim, &obj_inv_cov, &obj_entries, &obj_evals, &obj_w_norm, &obj_preFac, &det_inv_cov, &h) )
		return NULL;

	////////// GET FIRST LIST-GROUP FROM PYTHON ////////////
	ndata = PyList_Size(obj_w_norm);
	neval = PyList_Size(obj_evals) / ndim ;

	x = (double*) malloc(ndim * sizeof(double)*ndata);
	w_norm = (double*) malloc(sizeof(double)*ndata);
	preFac = (double*) malloc(sizeof(double)*ndata);
	evals = (double*) malloc(sizeof(double)*neval*ndim);

	nelems_c= PyList_Size(obj_inv_cov);
	inv_cov = (double*) malloc(sizeof(double)*nelems_c);

	// get data //
	for( i=0; i < ndim*ndata; i++ ) {
		ListItem = PyList_GetItem(obj_entries , i);

		if( PyFloat_Check(ListItem) ) {
			x[i] = PyFloat_AsDouble(ListItem);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	// get evals //
	for( i=0; i < neval*ndim; i++ ) {
		ListItem = PyList_GetItem(obj_evals , i);

		if( PyFloat_Check(ListItem) ) {
			evals[i] = PyFloat_AsDouble(ListItem);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	// get inv_cov //
	for( i=0; i < nelems_c; i++ ) {
		ListItem = PyList_GetItem(obj_inv_cov , i);

		if( PyFloat_Check(ListItem) ) {
			inv_cov[i] = PyFloat_AsDouble(ListItem);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}


	// get preFacs and weights //
	for( i=0; i < ndata; i++ ) {
		ListItem = PyList_GetItem(obj_preFac, i);
		ListItem2 = PyList_GetItem(obj_w_norm, i);

		if( PyFloat_Check(ListItem) && PyFloat_Check(ListItem2)) {
			preFac[i] = PyFloat_AsDouble(ListItem);
			w_norm[i] = PyFloat_AsDouble(ListItem2);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	// new stuff for ND //
	PyObject *pylist;
	pylist = PyList_New(neval);

	for( j = 0; j<neval;j++){
		double thiskde = 0.0;

		for( i = 0; i<ndata;i++){
			double exponent = 0.0;

			for( d = 0; d<ndim;d++){
				double inner = 0.0;

				for( k = 0;k<ndim;k++){
					inner += inv_cov[k+d*ndim] * (x[i+k*ndata] - evals[j+k*neval]);
				}
				exponent += inner * (x[i+d*ndata] - evals[j+d*neval]);
			}
			thiskde += w_norm[i] * exp(preFac[i] * exponent);
		}
		PyList_SET_ITEM(pylist, j, PyFloat_FromDouble(thiskde));
	}

	///////// FREE MEMORY ///////////
	free(x);
	free(preFac);
	free(w_norm);
	free(inv_cov);
	free(evals);

	///////// RETURN RESULTING VALUES FOR y /////////
	return pylist;
}


static PyObject *pr_getLambda_ND(PyObject *self, PyObject *args){
	////////////////// DECLARATIONS ///////////////////////
	int ndata, nelems_c, ndim, i,j,d,k;
	double invGlob, logSum, alpha, h, preFac, det_inv_cov;
	PyObject *obj_entries, *obj_w_norm, *obj_inv_cov;
	PyObject *ListItem, *ListItem_w;
	double *x, *lambda, *kde, *w_norm, *inv_cov;

	/////////////////// GET INPUT //////////////////////////
	if (!PyArg_ParseTuple(args, "iOOOddd", &ndim, &obj_inv_cov, &obj_entries, &obj_w_norm, &det_inv_cov, &h, &alpha ) )
		return NULL;

	////////// GET FIRST LIST-GROUP FROM PYTHON ////////////
	ndata = PyList_Size(obj_w_norm);
	x = (double*) malloc(ndim * sizeof(double)*ndata);
	lambda = (double*) malloc(sizeof(double)*ndata);
	kde = (double*) malloc(sizeof(double)*ndata);
	w_norm = (double*) malloc(sizeof(double)*ndata);

	nelems_c= PyList_Size(obj_inv_cov);
	inv_cov = (double*) malloc(sizeof(double)*nelems_c);

	for( i=0; i < ndata; i++ ) {
		ListItem_w = PyList_GetItem(obj_w_norm , i);
		if( PyFloat_Check(ListItem_w)) {
			w_norm[i] = PyFloat_AsDouble(ListItem_w);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	for( i=0; i < ndim*ndata; i++ ) {
		ListItem = PyList_GetItem(obj_entries , i);

		if( PyFloat_Check(ListItem) && PyFloat_Check(ListItem_w)) {
			x[i] = PyFloat_AsDouble(ListItem);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	for( i=0; i < nelems_c; i++ ) {
		ListItem = PyList_GetItem(obj_inv_cov, i);
		if( PyFloat_Check(ListItem)) {
			inv_cov[i] = PyFloat_AsDouble(ListItem);
		} else {
			printf("Error: lists contain a non-float value.\n");
			exit(1);
		}
	}

	/////////////// RUN CALCULATIONS /////////////////////
	PyObject *lambdaList;
	lambdaList = PyList_New(ndata);

	invGlob = 0.0;
	logSum = 0.0;
	preFac = -0.5/pow(h, 2);

	// new stuff for ND //
	for( j = 0; j<ndata;j++){
		double thiskde = 0.0;

		for( i = 0; i<ndata;i++){
			double exponent = 0.0;

			for( d = 0; d<ndim;d++){
				double inner = 0.0;

				for( k = 0;k<ndim;k++){
					inner += inv_cov[k+d*ndim] * (x[i+k*ndata] - x[j+k*ndata]);
				}
				exponent += inner * (x[i+d*ndata] - x[j+d*ndata]);
			}
			thiskde += w_norm[i] * exp(preFac * exponent);
		}
		kde[j] = thiskde;
		logSum += 1.0/ndata * log(thiskde);
	}

	invGlob = 1.0/exp(logSum);

	for( i=0; i< ndata; i++) {
		lambda[i] = 1.0/pow(invGlob*kde[i], alpha);
		PyList_SET_ITEM(lambdaList, i, PyFloat_FromDouble(lambda[i]));
	}

	///////// FREE MEMORY ///////////
	free(x);
	free(lambda);
	free(kde);
	free(w_norm);
	free(inv_cov);

	///////// RETURN RESULTING VALUES FOR LAMBDAS /////////
	return lambdaList;
}

static PyMethodDef PrMethods[] = {
	{"kde_2d", pr_kde_2d, METH_VARARGS, "Use Kde Calculation in 2D"},
	{"getLambda_2d", pr_getLambda_2d, METH_VARARGS, "Calculate lambdas for Kde Calculation in 2D"},
	{"kde_1d", pr_kde_1d, METH_VARARGS, "Use Kde Calculation in 1D"},
	{"getLambda_1d", pr_getLambda_1d, METH_VARARGS, "Calculate lambdas for Kde Calculation in 1D"},
	{"kde_ND", pr_kde_ND, METH_VARARGS, "Use ND"},
	{"getLambda_ND", pr_getLambda_ND, METH_VARARGS, "Calculate lambdas ND"},
	{NULL, NULL, 0, NULL}
};

void initkde(void){
	(void) Py_InitModule("kde", PrMethods);
}
