This is just a website, your can use any web server, just serve all the content under **client/web**.

The client is built as a single executable file.

```
$ client.exe
```

For Linux and Mac, or other Unix, the server can be built with:

```
$ go get -u github.com/gobuffalo/packr/packr
$ go get github.com/gobuffalo/packr@v1.30.1
$ packr build
```

The web page will be available at: http://localhost:3333/

That's it!
