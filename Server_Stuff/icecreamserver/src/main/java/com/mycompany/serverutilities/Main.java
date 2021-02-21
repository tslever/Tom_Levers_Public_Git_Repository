
// Allows executable to find class Main and exceptions thrown by method main.
package com.mycompany.serverutilities;

// Imports classes.
import com.mycompany.serverutilities.clientcommunicationutilities.Server;
import com.
       mycompany.
       serverutilities.
       clientcommunicationutilities.
       IceCreamClientCommunication;
import com.
       mycompany.
       serverutilities.
       clientcommunicationutilities.
       SetMessageInterfacesException;
import com.
       mycompany.
       serverutilities.
       clientcommunicationutilities.
       StartServerListeningForMessagesException;
import com.mycompany.serverutilities.productutilities.IceCreamProductRetrieval;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import org.apache.commons.lang.StringUtils;

/**
 * Defines class Main that encapsulates static method main, the entry point of
 * this application.
 * @version 0.0
 * @author Tom Lever
 */
public class Main {
    
    /**
     * Defines method main of this application, which instantiates a server,
     * an ice cream product retrieval subsystem, and an ice cream client
     * communication subsystem based on the server and the retrieval subsystem,
     * and which sets the message interfaces of the ice cream client
     * communication subsystem and starts the server listening for messages.
     * @param args
     * @throws InvalidPortNumberException
     * @throws IOException
     * @throws SetMessageInterfacesException
     * @throws StartServerListeningForMessagesException
     */
    public static void main(String[] args)
        throws InvalidPortNumberException,
               IOException,
               SetMessageInterfacesException,
               StartServerListeningForMessagesException {
        
        // isNumeric returns true if argument is a non-negative integer.
        if ((args.length < 1) || (!StringUtils.isNumeric(args[0]))) {
            throw new InvalidPortNumberException(
                "First command-line argument has been found to not " +
                "exist or not be a non-negative integer.");
        }
        
        int portNumber = Integer.parseInt(args[0]);
        
        HttpServer httpServer = HttpServer.create(
            new InetSocketAddress(portNumber), 0);
        
        Server server = new Server(httpServer);
        
        IceCreamProductRetrieval retriever = new IceCreamProductRetrieval();
        
        IceCreamClientCommunication communicator =
            new IceCreamClientCommunication(server, retriever);
        
        communicator.setMessageInterfaces();
        
        communicator.startServerListeningForMessages();
    }
}
