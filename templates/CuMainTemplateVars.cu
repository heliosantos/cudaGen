$!MEASURE_DECLARE_TIMER!$
	cudaEvent_t start, stop;
	float elapsedTime;
$!MEASURE_CREATE_TIMER!$
	/* create the timers */
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	/* start the timer */
	HANDLE_ERROR(cudaEventRecord(start, 0));
$!MEASURE_TERMINATE_TIMER!$
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	cudaEventSynchronize(stop);
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf(\"execution took %3.6f miliseconds\", elapsedTime);
$!MEASURE_FREE_TIMER!$
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
