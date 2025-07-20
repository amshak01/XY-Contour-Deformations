#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <complex.h>
#include <math.h>

#define SIZE 64
#define _USE_MATH_DEFINES
#define N_ITER 2e5

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct Node
{
    int index;
    struct Node *next;
} Node;

Node *createNode(int new_index)
{
    Node *new_node = (Node *)malloc(sizeof(Node));
    new_node->index = new_index;
    new_node->next = NULL;
    return new_node;
}

typedef struct Stack
{
    Node *head;
} Stack;

void initializeStack(Stack *stack) { stack->head = NULL; }

// Function to check if the stack is empty
int isEmpty(Stack *stack)
{

    // If head is NULL, the stack is empty
    return stack->head == NULL;
}

// Function to push an element onto the stack
void push(Stack *stack, int new_index)
{

    // Create a new node with given data
    Node *new_node = createNode(new_index);

    // Link the new node to the current top node
    new_node->next = stack->head;

    // Update the top to the new node
    stack->head = new_node;
}

// Function to remove the top element from the stack
int pop(Stack *stack)
{

    // Assign the current top to a temporary variable and read its value
    Node *temp = stack->head;
    int index = temp->index;

    // Update the top to the next node
    stack->head = stack->head->next;

    // Deallocate the memory of the old top node
    free(temp);

    // Return the value of the top element
    return index;
}

int wrap(int ind)
{
    if (ind < 0)
    {
        return SIZE - ((-ind) % SIZE);
    }
    else if (ind >= SIZE)
    {
        return ind % SIZE;
    }
    return ind;
}

double print_cluster(int *marks)
{
    int i;
    int j;

    for (i = 0; i < SIZE; i++)
    {
        for (j = 0; j < SIZE; j++)
        {
            printf("%d\t", marks[i * SIZE + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

double print_lattice(double *lat, FILE *lat_file)
{
    int i;
    int j;

    for (i = 0; i < SIZE; i++)
    {
        for (j = 0; j < SIZE; j++)
        {
            fprintf(lat_file, "%.4f", lat[i * SIZE + j]);
            if (j < SIZE - 1)
                fprintf(lat_file, ",");
        }
        fprintf(lat_file, "\n");
    }
}

int obs(double *lat, FILE *obs_file, FILE *corr_file)
{
    int i, j;
    double tot_energy = 0;

    for (i = 0; i < SIZE; i++)
    {
        complex row_tot = 0; // There are the Phi(t) we discussed

        for (j = 0; j < SIZE; j++)
        {
            row_tot += cexp(I * lat[i * SIZE + j]);

            double diff_right = lat[i * SIZE + j] - lat[i * SIZE + wrap(j + 1)];
            double diff_down = lat[i * SIZE + j] - lat[wrap(i + 1) * SIZE + j];

            tot_energy += (1 - cos(diff_right));
            tot_energy += (1 - cos(diff_down));
        }

        fprintf(corr_file, "%f%+fj", creal(row_tot), cimag(row_tot));
        if (i < SIZE - 1)
            fprintf(corr_file, ",");
    }
    fprintf(corr_file, "\n");

    // double energy = tot_energy/(SIZE*SIZE);
    fprintf(obs_file, "%f\n", tot_energy);

    return 0;
}

int simulate(double temp, char filename[])
{

    srand(time(NULL));

    // char obs_filename[20];
    // char corr_filename[20];
    char lat_filename[64];
    // strcpy(obs_filename, filename);
    // strcpy(corr_filename, filename);
    strcpy(lat_filename, filename);

    // FILE *obs_file;
    // obs_file = fopen(strcat(obs_filename, "_obs.csv"), "w+");

    // FILE *corr_file;
    // corr_file = fopen(strcat(corr_filename, "_corrs.csv"), "w+");

    FILE* lat_file;
    snprintf(lat_filename, sizeof(lat_filename), "%s_configs.bin", filename);
    lat_file = fopen(lat_filename, "wb");

    // allocate array of rows for lattice and markers
    double *lat = (double *)malloc(SIZE * SIZE * sizeof(double));
    int *marks = (int *)malloc(SIZE * SIZE * sizeof(int));

    int i, j;

    memset(marks, 0, SIZE * SIZE * sizeof(int));

    for (i = 0; i < SIZE; i++)
    {
        for (j = 0; j < SIZE; j++)
        {
            // give every site a random angle between 0 and 360 deg
            lat[i * SIZE + j] = ((double)rand() / RAND_MAX) * 2 * M_PI;
        }
    }

    int iter = 0;
    double sweeps = 0;

    // Create stack for cluster formation
    Stack *index_stack = malloc(sizeof(Stack));
    initializeStack(index_stack);

    // Create stack for cluster erasure
    Stack *marked_stack = malloc(sizeof(Stack));
    initializeStack(marked_stack);

    int calc_step = 10;
    double cluster_tot = 0;

    // Monte Carlo outer loop
    while (iter < N_ITER)
    {

        int cluster_size = 0;

        // choose random angle
        // note, unlike Wolff's original paper this is not the angle of the vector normal to the plane,
        // it is the angle the plane makes with the positive x axis
        double ref_angle = ((double)rand() / RAND_MAX) * 2 * M_PI;

        // choose a random site and push it to the stack
        int rand_site = rand() % (SIZE * SIZE);
        push(index_stack, rand_site);

        while (!isEmpty(index_stack))
        {

            // grab next site from stack, store its angle
            int site = pop(index_stack);
            double site_angle = lat[site];

            // skip iteration if site already marked
            if (marks[site])
            {
                continue;
            }

            // Flip the spin
            double new_angle = fmod(2 * ref_angle - site_angle, 2 * M_PI);
            lat[site] = new_angle < 0 ? 2 * M_PI + new_angle : new_angle; // wrap spin around to [0, 2*PI]

            // mark the site and push the site to the marked stack
            marks[site] = 1;
            push(marked_stack, site);
            cluster_size++;

            // turn 1d index into row and column for later
            int site_row = site / SIZE;
            int site_col = site % SIZE;

            // Find the 4 nearest neighbors to loop over (periodic boundary conditions)
            int neighbours[4] =
                {
                    wrap(site_row + 1) * SIZE + site_col,
                    wrap(site_row - 1) * SIZE + site_col,
                    site_row * SIZE + wrap(site_col + 1),
                    site_row * SIZE + wrap(site_col - 1)};

            int n;

            // loop over neighbours
            for (n = 0; n < 4; n++)
            {
                // get neighbouring spin angle
                double neighbour_angle = lat[neighbours[n]];

                // Wolff probability (differs from Tej's code by a minus sign since I'm flipping during cluster formation instead of after)
                //  the cosines from the Wolff paper become sines because this angle is offset by -pi/2
                double p = 2 * sin(lat[site] - ref_angle) * sin(neighbour_angle - ref_angle) / temp;
                p = p < 0 ? p : 0;
                double prob = 1 - exp(p);

                // generate random uniform on [0,1) to compare to prob
                double thresh = (double)rand() / RAND_MAX;

                if (prob > thresh)
                    push(index_stack, neighbours[n]);
            }
        }
        // don't calculate observables on every step
        if (iter % calc_step == 0 && iter >= N_ITER / 10)
        {
            fwrite(lat, sizeof(double), SIZE*SIZE, lat_file);
        }

        // unmark the spins in a smart way, we push the indices of the marked spins to a stack
        while (!isEmpty(marked_stack))
            marks[pop(marked_stack)] = 0;

        iter++;
        cluster_tot += (double)cluster_size;
        sweeps += (double)cluster_size / (SIZE * SIZE);
    }

    free(index_stack);
    free(marked_stack);
    free(marks);
    free(lat);

    // fclose(obs_file);
    // fclose(corr_file);
    fclose(lat_file);

    // printf("Average Cluster Size of %.5f\nTotal updates: %d\n", (cluster_tot / iter) / (SIZE * SIZE), iter);

    return iter;
}

int main(void)
{

    double min_temp = 0.8;
    double max_temp = 1.0;
    int n_temps = 20;

    printf("Beginning simulation of XY model on a %dx%d lattice...\n", SIZE, SIZE);

    // FILE* runtimes;
    // runtimes = fopen("configs/runtimes.txt", "w+");

    int i;
    for (i = 0; i <= n_temps; i++)
    {
        double temp = min_temp + i * (max_temp - min_temp) / n_temps;
        char write_to[64];
        sprintf(write_to, "test_configs/L=%d_cluster_T=%1.4f", SIZE, temp);
        clock_t begin = clock();
        int iter = simulate(temp, write_to);
        clock_t end = clock();
        double dt = (double)(end - begin)/CLOCKS_PER_SEC;
        // fprintf(runtimes, "%.4f\t%.4f\t%d\n", temp, dt, iter);
    }

    // fclose(runtimes);

    // double temp = 1.0;

    // char write_to[20];
    // sprintf(write_to, "configs/L=%d_cluster_T=%1.4f", SIZE, temp);
    // printf(write_to);
    // printf("\n");
    // simulate(temp, write_to);

    return 0;
}