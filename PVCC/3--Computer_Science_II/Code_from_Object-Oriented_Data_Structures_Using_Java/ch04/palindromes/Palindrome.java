//---------------------------------------------------------------------
// Palindrome.java           by Dale/Joyce/Weems              Chapter 4
//
// Provides a method to test whether a string is a palindrome. 
// Non letters are skipped.
//---------------------------------------------------------------------
package ch04.palindromes;

import ch02.stacks.*;
import ch04.queues.*;

public class Palindrome 
{
  public static boolean test(String candidate)
  // Returns true if candidate is a palindrome, false otherwise.
  {
    char ch;                   // current candidate character being processed
    int length;                // length of candidate string
    char fromStack;            // current character popped from stack
    char fromQueue;            // current character dequeued from queue
    boolean stillPalindrome;   // true if string might still be a palindrome

    StackInterface<Character> stack;   // holds non blank string characters
    QueueInterface<Character> queue;   // also holds non blank string characters

    // initialize variables and structures
    length = candidate.length();
    stack = new ArrayBoundedStack<Character>(length);
    queue = new ArrayBoundedQueue<Character>(length);

    // obtain and handle characters
    for (int i = 0; i < length; i++)
    {
      ch = candidate.charAt(i);
      if (Character.isLetter(ch))
      {
        ch = Character.toLowerCase(ch);
        stack.push(ch);
        queue.enqueue(ch);
      }
    }
    
    // determine if palindrome
    stillPalindrome = true;
    while (stillPalindrome && !stack.isEmpty())
    {
      fromStack = stack.top();
      stack.pop();
      fromQueue = queue.dequeue();
      if (fromStack != fromQueue)
        stillPalindrome = false;
    }
    
    // return result
    return stillPalindrome;
  }
}