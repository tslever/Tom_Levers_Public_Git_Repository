package Com.TSL.LineEditorUtilities;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;


/** 
* LineList is a linked-based list that represents the contents of a document.
*
* Student name: Tom Lever
* Completion date: 06/27/21
*/

public class LineList {


    private Node<String> head;

    
    /**
     * LineList is the zero-parameter constructor for LineList, which sets this list's reference to its head node to
     * null.
     */

    public LineList() {
 
        head = null;

    }


    /**
     * addLine(String line) appends a provided line to the end of this list of lines.
     * 
     * @param line
     */
    
    public void addLine(String line) throws AnInsertsStringException, AnInvalidCharacterException {

    	if (line == null) {
    		throw new AnInsertsStringException(
    			"The line editor found that a reference to a string to append was null.\n"
    		);
    	}
    	
    	if (line.equals("")) {
    		throw new AnInsertsStringException(
    			"The line editor found that a reference to a string to append was empty.\n"
    		);
    	}
    	
    	for (int i = 0; i < line.length(); i++) {
    		check(line.charAt(i));
    	}
		
		Node<String> theSinglyLinkedListNodeForTheLine = new Node<String>(line, null);
		
		if (this.head == null) {
			this.head = theSinglyLinkedListNodeForTheLine;
			return;
		}
		
		Node<String> theCurrentSinglyLinkedListNode = this.head;
		while (theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode() != null) {
			theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
		}
		
		theCurrentSinglyLinkedListNode.setsItsReferenceToTheNextNodeTo(theSinglyLinkedListNodeForTheLine);
    	
    }
    
    
    /**
     * addLine(String line, int n) inserts a provided line at a provided index.
     * 
     * @param line
     * @param n
     * @throws AnInsertsStringException
     * @throws AnInvalidCharacterException
     */
    
    public void addLine(String line, int n) throws AnInsertsStringException, AnInvalidCharacterException {
    	
    	n -= 1;
    	
    	if (line == null) {
    		throw new AnInsertsStringException(
    			"The line editor found that a reference to a string to append was null.\n"
    		);
    	}
    	
    	if (line.equals("")) {
    		throw new AnInsertsStringException(
    			"The line editor found that a reference to a string to append was empty.\n"
    		);
    	}
    	
		if (n < 0) {
			throw new AnInsertsStringException(
				"The line editor found that an index to use to insert a line in the line editor's buffer of strings " +
				"was negative.\n"
			);
		}
    	
    	for (int i = 0; i < line.length(); i++) {
    		check(line.charAt(i));
    	}
    	
    	Node<String> theSinglyLinkedListNodeForTheLine = new Node<String>(line, null);
    	
		if (this.head == null) {
			this.head = theSinglyLinkedListNodeForTheLine;
			return;
		}
		
		if (n == 0) {
			theSinglyLinkedListNodeForTheLine.setsItsReferenceToTheNextNodeTo(this.head);
			this.head = theSinglyLinkedListNodeForTheLine;
			return;
		}
		
		if (n >= lines()) {
			
			addLine(line);
			
			System.out.print("The line editor appended \"" + line + "\" to its buffer of strings.\n\n");
			
		}
		
		else {
					
			Node<String> thePreviousSinglyLinkedListNode = this.head;
			Node<String> theCurrentSinglyLinkedListNode = this.head.providesItsReferenceToTheNextNode();
			int theIndexOfTheCurrentNode = 1;
			
			while ((theCurrentSinglyLinkedListNode != null) && (theIndexOfTheCurrentNode < n)) {
				thePreviousSinglyLinkedListNode = theCurrentSinglyLinkedListNode;
				theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
				theIndexOfTheCurrentNode++;
			}
		
			n += 1;
			
			System.out.print(
				"The line editor inserted " + theSinglyLinkedListNodeForTheLine.providesItsData() +
				" into its buffer of strings at index " + n + ".\n\n"
			);
			
			thePreviousSinglyLinkedListNode.setsItsReferenceToTheNextNodeTo(theSinglyLinkedListNodeForTheLine);
			theSinglyLinkedListNodeForTheLine.setsItsReferenceToTheNextNodeTo(theCurrentSinglyLinkedListNode);
			
		}
    	
    }
    
    
    /**
     * check throws an invalid character exception if the provided character is not a tab, newline character, carriage
     * return, space, visible ASCII character other than space and backslash, left single quote, and right single quote.
     * 
     * @param thePresentCharacter
     * @throws AnInvalidCharacterException
     */
    
    private void check(char thePresentCharacter) throws AnInvalidCharacterException {
    		
		if ((thePresentCharacter != '\t') &&
			(thePresentCharacter != '\n') &&
			(thePresentCharacter != '\r') &&
			!((thePresentCharacter >= ' ') && (thePresentCharacter <= '[')) &&
			!((thePresentCharacter >= ']') && (thePresentCharacter <= '~')) &&
			(thePresentCharacter != '\u2018') &&
			(thePresentCharacter != '\u2019')
		   ) {
			throw new AnInvalidCharacterException(
				"The UTF-8 character with index " + (int)thePresentCharacter + " is invalid."
			);
		}
		
    }
    
    
    /**
     * delete deletes the line in the line editor's buffer of strings with index n, or outputs a warning that the
     * provided index does not correspond to a line in the buffer of strings.
     * 
     * @param n
     */
    
    public void delete(int n) {
    	
    	n -= 1;
    	
    	if ((n < 0) || (n >= lines())) {
    		System.out.print("Line " + n + " does not exist in the line editor's buffer of strings.\n\n");
    		return;
    	}
    	
    	if (n == 0) {
    		this.head = this.head.providesItsReferenceToTheNextNode();
    		return;
    	}
    	
    	Node<String> thePreviousSinglyLinkedListNode = this.head;
    	Node<String> theCurrentSinglyLinkedListNode = this.head.providesItsReferenceToTheNextNode();
    	int theIndexOfTheCurrentNode = 1;
    	
    	while (theIndexOfTheCurrentNode < n) {
    		thePreviousSinglyLinkedListNode = theCurrentSinglyLinkedListNode;
    		theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
    		theIndexOfTheCurrentNode++;
    	}
    	
    	thePreviousSinglyLinkedListNode.setsItsReferenceToTheNextNodeTo(
    		theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode()
    	);
    	
    	n += 1;
    	
    	// TODO: Uncomment after testing.
    	//System.out.print(
    	//	"The line editor removed line " + n + ": " + theCurrentSinglyLinkedListNode.providesItsData() + ".\n\n"
    	//);
    	
    }
    
    
    /**
     * empty clears this list of all lines.
     */
    
    public void empty() {
    	
    	this.head = null;
    	
		System.out.print("The line editor's buffer of strings was cleared.\n\n");
    	
    }
    
    
    /**
     * line outputs the line with a provided index to the standard output stream.
     * 
     * @param n
     */
    
    public void line(int n) {
    	
    	n -= 1;
    	
    	if ((n < 0) || (n >= lines())) {
    		System.out.println("Line " + (n+1) + " does not exist.");
    		return;
    	}
    	
    	Node<String> theCurrentSinglyLinkedListNode = this.head;
    	int theIndexOfTheCurrentNode = 0;
    	while (theIndexOfTheCurrentNode < n) {
    		theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
    		theIndexOfTheCurrentNode++;
    	}
    	
    	System.out.println(
    		theCurrentSinglyLinkedListNode.providesItsData()
    		.replace("\t", "\\t")
    		.replace("\n", "\\n")
    		.replace("\r", "\\r")
    	);

    }
    
    
    /**
     * lines provides the number of lines in this list.
     * 
     * @return
     */
    
    public int lines() {
    	
    	int theNumberOfLines = 0;
    	
    	Node<String> theCurrentSinglyLinkedListNode = this.head;
    	while (theCurrentSinglyLinkedListNode != null) {
    		theNumberOfLines++;
    		theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
    	}
    	
    	return theNumberOfLines;
    	
    }
    
    
    /**
     * load clears the line editor's buffer of strings if a append / overwrite indicate is false, loads the file at the
     * provided path, and appends the lines in that file to the line editor's buffer of strings.
     * 
     * @param fileName
     * @param append
     */
    
    public void load(String fileName, boolean append) {

    	if (!append) {
    		empty();
    	}
    	
    	try {
    		loadsTheFileAt(fileName);
    	}
    	catch (IOException theIOException) {
    		System.out.println(theIOException.getMessage() + "\n");
    	}
    	catch (AnInsertsStringException theInsertsStringException) {
    		System.out.println(theInsertsStringException.getMessage());
    	}
    	catch (AnInvalidCharacterException theInvalidCharacterException) {
    		System.out.println(theInvalidCharacterException.getMessage() + "\n");
    	}
    	
    }
    
    
	/**
	 * loadsTheFileAt appends the lines of the file at a provided path into the line editor's buffer of strings.
	 * 
	 * @param args
	 * @throws AnInvalidCharacterException
	 * @throws IOException
	 */
	
	public void loadsTheFileAt(String thePath)
		throws AnInsertsStringException, AnInvalidCharacterException, IOException {		
		
		File theFile = new File(thePath);
		int theNumberOfLinesInTheFile = 0;
		FileReader theFileReader = new FileReader(theFile, StandardCharsets.UTF_8);
		BufferedReader theBufferedReader = new BufferedReader(theFileReader);
		
		int thePresentCharacterAsAnInteger;
		char thePresentCharacter;
		StringBuilder theStringBuilder = new StringBuilder();
		while ((thePresentCharacterAsAnInteger = theBufferedReader.read()) != -1) {
			
			thePresentCharacter = (char)thePresentCharacterAsAnInteger;
			
			check(thePresentCharacter);
			
			if (thePresentCharacter != '\n') {
				theStringBuilder.append(thePresentCharacter);
			}
			else {
				LineEditor.bufferOfStrings.addLine(theStringBuilder.toString());
				theNumberOfLinesInTheFile++;
				theStringBuilder = new StringBuilder();
			}
		}
		if (theStringBuilder.length() != 0) {
			LineEditor.bufferOfStrings.addLine(theStringBuilder.toString());
			theNumberOfLinesInTheFile++;
		}
		
		System.out.println(
			"The line editor added to its buffer of strings " + theNumberOfLinesInTheFile +
			" lines from the file at \"" + thePath + "\".\n"
		);
		
		theBufferedReader.close();
		theFileReader.close();
		
	}
    
    
    /**
     * print outputs the lines in this list, along with their indices, to the standard output stream.
     */
    
    public void print() {
    	
    	if (this.head == null) {
    		return;
    	}
    	
	    Node<String> theCurrentSinglyLinkedListNode = this.head;
	    int theIndexOfTheCurrentLine = 1;
		
		String theCurrentLine;
		while (theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode() != null) {
			theCurrentLine = theCurrentSinglyLinkedListNode.providesItsData();
			System.out.println(
				theIndexOfTheCurrentLine + ". " +
				theCurrentLine.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
			);
			theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
			theIndexOfTheCurrentLine++;
		}
		
		System.out.println(
			theIndexOfTheCurrentLine + ". " +
			theCurrentSinglyLinkedListNode.providesItsData()
			.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
		);
		
		//System.out.println(); // TODO: Uncomment after testing.

    }
    
    
    /**
     * replace replaces all occurrences in each line in the line editor's buffer of strings of a first provided string
     * with a second provided string.
     * 
     * @return
     */
    
    public void replace(String s1, String s2) {
    	
	    Node<String> theCurrentSinglyLinkedListNode = this.head;
	    
		while (theCurrentSinglyLinkedListNode != null) {
			theCurrentSinglyLinkedListNode.setsItsDataTo(
				theCurrentSinglyLinkedListNode.providesItsData().replace(s1, s2)
			);
			
			theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
		}
		
		// TODO: Uncomment after testing.
		//System.out.print(
		//	"All instances of \"" + s1 + "\" in the line editor's buffer of strings were replaced with \"" + s2 +
		//	"\".\n\n"
		//);

    }
    
    
    /**
     * save saves the lines in this list to a file at a provided path.
     * 
     * @param fileName
     */
    
    public void save(String fileName) {
    	
    	FileWriter theFileWriter;
    	try {
    		theFileWriter = new FileWriter(fileName);
    	}
    	catch (IOException theIOException) {
    		System.out.println(
    			"The line editor was unable to save the lines in its buffer of strings because the file specified " +
    			"by the path that the line editor received could not be opened."
    		);
    		return;
    	}
    	
    	BufferedWriter theBufferedWriter = new BufferedWriter(theFileWriter);
    	
    	if (this.head != null) {
    	
	    	Node<String> theCurrentSinglyLinkedListNode = this.head;
	    	String theCurrentLine;
	    	
	    	while (theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode() != null) {
	    		theCurrentLine = theCurrentSinglyLinkedListNode.providesItsData();
	    		if (theCurrentLine.charAt(theCurrentLine.length() - 1) != '\n') {
	    			theCurrentLine += "\n";
	    		}
	    		try {
	    			theBufferedWriter.write(theCurrentLine);
	    		}
	    		catch (IOException theIOException) {
	    			System.out.println("The line editor could not write the line \"" + theCurrentLine + "\".");
	    		}
	    		theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
	    	}
	    	
	    	theCurrentLine = theCurrentSinglyLinkedListNode.providesItsData();
    		try {
    			theBufferedWriter.write(theCurrentLine);
    		}
    		catch (IOException theIOException) {
    			System.out.println("The line editor could not write the line \"" + theCurrentLine + "\".");
    		}
    		theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
    	
    	}
    	
    	try {
    		theBufferedWriter.flush();
    	}
    	catch (IOException theIOException) {
    		System.out.println("The line editor could not flush a buffered writer.");
    	}
    	
    	try {
    		theBufferedWriter.close();
    	}
    	catch (IOException theIOException) {
    		System.out.println("The line editor could not close a buffered writer.");
    	}
    	
    	try {
    		theFileWriter.close();
    	}
    	catch (IOException theIOException) {
    		System.out.println("The line editor could not close a file writer.");
    	}
    	
		System.out.print(
			"The lines in the line editor's buffer of strings were saved to a file at " + fileName + ".\n\n"
		);

    }
    
    
    /**
     * toString provides a string representation of this list.
     * 
     * @return
     */
    
    public String toString() {
    	
    	if (this.head == null) {
    		return "";
    	}
    	
    	String theRepresentationOfThisList = "";
    	
	    Node<String> theCurrentSinglyLinkedListNode = this.head;
		while (theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode() != null) {
			theRepresentationOfThisList += theCurrentSinglyLinkedListNode.providesItsData() + "\n";
			theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
		}
		
		theRepresentationOfThisList += theCurrentSinglyLinkedListNode.providesItsData();
		
		return theRepresentationOfThisList;

    }
    
    
    /**
     * words provides the number of words in all of the lines in the line editor's buffer of strings.
     * 
     * @return
     */
    
    public int words() {
    	
    	int theNumberOfWords = 0;
    	
    	Node<String> theCurrentSinglyLinkedListNode = this.head;
    	Scanner theScannerForTheLine;
    	while (theCurrentSinglyLinkedListNode != null) {
    		
    		theScannerForTheLine = new Scanner(theCurrentSinglyLinkedListNode.providesItsData());
    		theScannerForTheLine.useDelimiter("[\\s\t,.;\u2018\u2019?*!\"@\\-:]+");
    		while (theScannerForTheLine.hasNext()) {
    			theNumberOfWords++;
    			theScannerForTheLine.next();
    		}
    		theCurrentSinglyLinkedListNode = theCurrentSinglyLinkedListNode.providesItsReferenceToTheNextNode();
    		
    	}
    	
    	return theNumberOfWords;
    	
    }
    
}