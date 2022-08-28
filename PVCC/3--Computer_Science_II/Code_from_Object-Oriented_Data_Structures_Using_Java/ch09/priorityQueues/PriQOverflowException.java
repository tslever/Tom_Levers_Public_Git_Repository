package ch09.priorityQueues;

class PriQOverflowException extends RuntimeException
{
  public PriQOverflowException()
  {
    super();
  }

  public PriQOverflowException(String message)
  {
    super(message);
  }
}