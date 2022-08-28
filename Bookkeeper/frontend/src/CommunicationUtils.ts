function readResponseStreamToCompletion (response: Response): Promise<string> {
    return response.text();
}

export default readResponseStreamToCompletion;