#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <immintrin.h>
#include <time.h>

#include "pagerank.h"

bool timer = 0;
clock_t tick;
clock_t tock;

typedef struct {
	size_t id;
	double *array;
	double *result;
	double dampener;
	size_t npages;
	size_t nthreads;
} m_hat_args;

void* calc_m_hat(void* args) {

	m_hat_args* wargs = (m_hat_args*) args;
	size_t length = wargs->npages * wargs->npages;
	size_t chunk = length / wargs->nthreads;
	const size_t start = wargs->id * chunk;
	const size_t end = wargs->id == wargs->nthreads - 1 ? length : (wargs->id + 1) * chunk;

	// Calculate M hat.
	for (int i = start; i < end; i++) {
		wargs->result[i] = wargs->dampener * wargs->array[i] + ((1.0 - wargs->dampener) / wargs->npages);
	}

	return NULL;
}

typedef struct {
	size_t id;
	double *one;
	double *two;
	double *three;
	size_t npages;
	size_t nthreads;
} scores_args;


void* calc_scores(void* args) {

	scores_args* wargs = (scores_args*) args;
	size_t length = wargs->npages;
	size_t chunk = length / wargs->nthreads;
	const size_t start = wargs->id * chunk;
	const size_t end = wargs->id == wargs->nthreads - 1 ? length : (wargs->id + 1) * chunk;

	double result1 = 0;
	double result2 = 0;
	double result3 = 0;
	double result4 = 0;

	// Calculate Scores
	for (int y = start; y < end; y++) {
		wargs->one[y] = 0.0;
		for (int x = 0; x < wargs->npages; x+=4) {
			result1 = wargs->two[y * wargs->npages + x] * wargs->three[x];
			result2 = wargs->two[y * wargs->npages + x+1] * wargs->three[x+1];
			result3 = wargs->two[y * wargs->npages + x+2] * wargs->three[x+2];
			result4 = wargs->two[y * wargs->npages + x+3] * wargs->three[x+3];
			wargs->one[y] += result1 + result2 + result3 + result4;
		}
	}

	return NULL;
}


typedef struct {
	size_t id;
	double *array;
	size_t npages;
	size_t nthreads;
} p_args;


void* init_matrixP(void* args) {

	p_args* wargs = (p_args*) args;
	size_t chunk = wargs->npages / wargs->nthreads;
	const size_t start = wargs->id * chunk;
	const size_t end = wargs->id == wargs->nthreads-1 ? wargs->npages : (wargs->id + 1) * chunk;

	for (size_t i = start; i < end; i++) {
		wargs->array[i] = 1.0 / wargs->npages;
	}

	return NULL;
}

bool j_links_i(node *inlinks, size_t pagej) {
	// Returns true if pagej is found in inlinks, false otherwise.
	for (node *inlink = inlinks; inlink != NULL; inlink = inlink->next) {
		if (inlink->page->index == pagej) {
			return true;
		}
	}
	return false;
}

bool check_convergence(double* a, double *b, size_t npages) {

	// Check whether the algorithm has converged.
	double result1 = 0;
	double result2 = 0;
	double result3 = 0;
	double result4 = 0;

	for (int i = 0; i < npages; i+=4) {
		result1 += (b[i] - a[i]) * (b[i] - a[i]);
		result2 += (b[i+1] - a[i+1]) * (b[i+1] - a[i+1]);
		result3 += (b[i+2] - a[i+2]) * (b[i+2] - a[i+2]);
		result4 += (b[i+3] - a[i+3]) * (b[i+3] - a[i+3]);
	}

	return sqrt(result1 + result2 + result3 + result4) <= EPSILON;
}

typedef struct {
	size_t id;
	double *result;
	node *pagei;
	node *pagej;
	node *list;
	size_t npages;
	size_t nthreads;
} m_args;


void* calc_m(void* args) {

	m_args* wargs = (m_args*) args;
	size_t length = wargs->npages;
	size_t chunk = length / wargs->nthreads;
	const size_t start = wargs->id * chunk;
	const size_t end = wargs->id == wargs->nthreads-1 ? length : (wargs->id + 1) * chunk;

	for (int i = 0; i < start; i++) {
		wargs->pagei = wargs->pagei->next;
	}

	// Calculate matrix M.
	for (int y = start; y < end; y++) {
		for (int x = 0; x < wargs->npages; x++) {

			if (wargs->pagej->page->noutlinks == 0)
				wargs->result[y * wargs->npages + x] = 1.0 / wargs->npages;
			else if (j_links_i(wargs->pagei->page->inlinks, wargs->pagej->page->index))
				wargs->result[y * wargs->npages + x] = 1.0 / wargs->pagej->page->noutlinks;
			else
				wargs->result[y * wargs->npages + x] = 0.0;

			wargs->pagej = wargs->pagej->next;
		}
		wargs->pagei = wargs->pagei->next;
		wargs->pagej = wargs->list;
	}

	return NULL;
}

void pagerank(node* list, size_t npages, size_t nedges, size_t nthreads, double dampener) {

	// Start clock.
	if (timer) tick = clock();

	double *matrixM = (double*)malloc(npages * npages * sizeof(double));

	// Thread calculation: matrix M.
	m_args argsM[nthreads];
	for (size_t i = 0; i < nthreads; i++) {
		argsM[i] = (m_args) {
			.id = i,
			.result = matrixM,
			.pagei = list,
			.pagej = list,
			.list = list,
			.npages = npages,
			.nthreads = nthreads
		};
	}
	pthread_t m_thread_ids[nthreads];
	// Launch threads
	for (size_t i = 0; i < nthreads; i++) {
		pthread_create(m_thread_ids + i, NULL, calc_m, argsM + i);
	}
	// Wait for threads to finish
	for (size_t i = 0; i < nthreads; i++) {
		pthread_join(m_thread_ids[i], NULL);
	}
	

	// Thread calculation: matrix M hat.
	double *matrixM_hat = calloc(npages * npages, sizeof(double));
	m_hat_args args[nthreads];
	for (size_t i = 0; i < nthreads; i++) {
		args[i] = (m_hat_args) {
			.id = i,
			.array = matrixM,
			.result = matrixM_hat,
			.dampener = dampener,
			.npages = npages,
			.nthreads = nthreads
		};
	}
	pthread_t thread_ids[nthreads];
	// Launch threads
	for (size_t i = 0; i < nthreads; i++) {
		pthread_create(thread_ids + i, NULL, calc_m_hat, args + i);
	}
	// Wait for threads to finish
	for (size_t i = 0; i < nthreads; i++) {
		pthread_join(thread_ids[i], NULL);
	}


	// Malloc matrix P for PageRank scores. Initialize to 1.0 / N.
	double *matrixP = malloc(npages * sizeof(double));

	// Thread calculation: matrix P
	p_args argsP[nthreads];
	for (size_t i = 0; i < nthreads; i++) {
		argsP[i] = (p_args) {
			.id = i,
			.array = matrixP,
			.npages = npages,
			.nthreads = nthreads
		};
	}
	pthread_t p_thread_ids[nthreads];
	// Launch threads
	for (size_t i = 0; i < nthreads; i++) {
		pthread_create(p_thread_ids + i, NULL, init_matrixP, argsP + i);
	}
	// Wait for threads to finish
	for (size_t i = 0; i < nthreads; i++) {
		pthread_join(p_thread_ids[i], NULL);
	}


	double *tmp = calloc(npages, sizeof(double));
	bool converged = false;

	while (!converged) {
		
		// Thread calculation: PageRank Scores
		scores_args argsS[nthreads];
		for (size_t i = 0; i < nthreads; i++) {
			argsS[i] = (scores_args) {
				.id = i,
				.one = tmp,
				.two = matrixM_hat,
				.three = matrixP,
				.npages = npages,
				.nthreads = nthreads
			};
		}
		pthread_t s_thread_ids[nthreads];
		// Launch threads
		for (size_t i = 0; i < nthreads; i++) {
			pthread_create(s_thread_ids + i, NULL, calc_scores, argsS + i);
		}
		// Wait for threads to finish
		for (size_t i = 0; i < nthreads; i++) {
			pthread_join(s_thread_ids[i], NULL);
		}

		if (check_convergence(matrixP, tmp, npages))
			converged = true;
		else
			memcpy(matrixP, tmp, npages * sizeof(double));

	}

	// Display page rank scores for each.
	for (int i = 0; i < npages; i++) {
		printf("%s %.8lf\n", list->page->name, tmp[i]);
		list = list->next;
	}

	free(matrixM);
	free(matrixM_hat);
	free(matrixP);
	free(tmp);

	// End clock.
	if (timer) {
		tock = clock();
		printf("time elapsed: %fs\n", (double) (tock - tick) / CLOCKS_PER_SEC);
	}

}

int main(int argc, char** argv) {
	config conf;
	init(&conf, argc, argv);

	node* list = conf.list;
	size_t npages = conf.npages;
	size_t nedges = conf.nedges;
	size_t nthreads = conf.nthreads;
	double dampener = conf.dampener;

	pagerank(list, npages, nedges, nthreads, dampener);
	release(list);
	return 0;
}
