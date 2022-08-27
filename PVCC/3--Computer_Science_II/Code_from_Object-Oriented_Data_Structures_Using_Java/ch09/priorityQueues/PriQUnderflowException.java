package ch09.priorityQueues;

class PriQUnderflowException extends RuntimeException
{
  public PriQUnderflowException()
  {
    super();
  }

  public PriQUnderflowException(String message)
  {
    super(message);
  }
}