
// Allows ControllerToProcessIntoAnswer to extend Controller.
package com.mycompany.serverutilities.clientcommunicationutilities;

// Imports classes.
import com.mycompany.serverutilities.productutilities.Answer;

/**
 * Defines abstract class Controller that declares abstract method process,
 * which when implemented will process the inputToProcess into an Answer.
 * @version 0.0
 * @author Tom Lever
 */
abstract class Controller {
    
    /**
     * Defines constructor Controller.
     */
    public Controller() { }
    
    /**
     * Declares abstract method process, which when implemented will process
     * the inputToProcess into an Answer.
     * @param messageToProcess
     */
    abstract public Answer process(Object inputToProcess)
        throws ProcessException;
}