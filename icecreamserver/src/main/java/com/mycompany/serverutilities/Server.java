
// Allows executable to find class Main.
package com.mycompany.serverutilities;

// Import classes.
import com.sun.net.httpserver.HttpServer;
//import java.io.BufferedReader;
//import java.io.InputStreamReader;
//import java.io.IOException;
//import java.io.PrintWriter;
//import java.net.Socket;

/**
 * Defines class Server that allows creation of a server based on an inputted
 * httpServer object that will allow for listening for HTTP requests from
 * clients and providing HTTP responses to clients.
 * The server's message interfaces will receive actually HTTP messages directed
 * to server endpoints and will provide HTTP messages to clients.
 * Server contains a method for setting a server's array of message interfaces
 * based on an inputted array of method interfaces, each with specific endpoints
 * and methods of processing HTTP requests from clients.
 * @version 0.0
 * @author Tom Lever
 */
public class Server {
    
    private final HttpServer httpServer;
    private MessageInterface[] messageInterfaces;
    private boolean setMessageInterfacesHasAlreadyBeenCalled;
    private boolean serverIsUnableToListenForMessages;
    
    
    /**
     * Constructs Server object based on inputted HttpServer object.
     * @version 0.0
     * @author Tom Lever
     */
    public Server(HttpServer httpServerToUse) {    
        this.httpServer = httpServerToUse;
        setMessageInterfacesHasAlreadyBeenCalled = false;
        serverIsUnableToListenForMessages = true;
    }
    
    /**
     * Defines Server's setMessageInterfaces based on an inputted array of
     * message interfaces.
     * @version 0.0
     * @author Tom Lever
     * @param messageInterfacesToUse
     * @throws Exception
     */
    public void setMessageInterfaces(
            MessageInterface[] messageInterfacesToUse) throws Exception {
        
        if (setMessageInterfacesHasAlreadyBeenCalled) {
            throw new Exception(
                "Message interfaces have already been set in Server.");
            // TODO: Create SetMessageInterfacesHasAlreadyBeenCalledException
            // class.
        }
        
        this.messageInterfaces = messageInterfacesToUse;
        
        // Call httpServer's createContext method for every MessageInterface.
        // After calling createContext, whenever a message with an endpoint
        // of <the result of message.getEndpoint()> is received by httpServer,
        // the method <handle> of the object
        // <the result of messageInterface.getProcessing()> will be called.
        for (MessageInterface messageInterface : this.messageInterfaces) {
            httpServer.createContext(
                messageInterface.getEndpoint(),
                messageInterface.getProcessing());
        }
        
        setMessageInterfacesHasAlreadyBeenCalled = true;
        serverIsUnableToListenForMessages = false;
    }
    
    public void startListeningForMessages() throws Exception {
        
        // If / else block is important because we don't want to
        // startListeningForMessages until setMessageInterfaces have been called,
        // because this.httpServer would not have processings for any endpoints.
        if (serverIsUnableToListenForMessages) {
            throw new Exception(
                "Server is unable to listen for messages: " +
                "message interfaces have not been created.");
            // TODO: Create ServerUnableToListenForMessagesException class.
        }
        this.httpServer.start();
    
    }
}