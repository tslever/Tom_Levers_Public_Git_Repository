package Com.TSL.UtilitiesForWorkingWithDoubleEndedQueues;


/**
* @author EMILIA BUTU
* version 1.0
* since   2020-03
*
* Student name: Tom Lever
* Completion date: 06/10/21
*
* LLNode.java: implements <T> nodes for a Linked List.
*/

public class LLNode<T>
{
	protected LLNode<T> link;
	protected T data;

	
	/**
	 * LLNode() is a zero-parameter constructor for LLNode, which sets this node's data and link variables to
	 * reference null.
	 */
	
	public LLNode()
	{
		data=null;
		link=null;
	}

	
	/**
	 * LLNode(T d) is a one-parameter constructor for LLNode, which sets this node's data to d and sets this node's
	 * link variable to reference null.
	 * 
	 * @param d
	 */
	
	public LLNode(T d)
	{
		data = d;
		link = null;
	}

	
	/**
	 * LLNode(T d, LLNode n) is a two-parameter constructor for LLNode, which sets this node's data to d and sets this
	 * node's link to n.
	 * 
	 * @param d
	 * @param n
	 */
	
	public LLNode(T d, LLNode n)
	{
		data=d;
		link=n;
	}

	
	/**
	 * setLink sets this node's link to link.
	 * 
	 * @param link
	 */
	
	public void setLink(LLNode<T> link)
	{
		this.link = link;
	}

	
	/**
	 * setData sets this node's data to d.
	 * 
	 * @param d
	 */
	
	public void setData(T d)
	{
	  	data = d;
	}

	
	/**
	 * getData gets this node's data.
	 * 
	 * @return
	 */
	
	public T getData()
	{
		return data;
	}

	
	/**
	 * getLink gets this node's link.
	 * 
	 * @return
	 */
	
	public LLNode<T> getLink()
	{
		return link;
	}
}