export async function fetchAccount(accountName: string): Promise<string[][]> {
    const promiseOfResponse = await fetch('http://localhost:3001/get_account?name=' + accountName);
    const accountAsString = await promiseOfResponse.json();
    return accountAsString;
}