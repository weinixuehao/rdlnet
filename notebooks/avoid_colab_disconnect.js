// 每60分钟自动运行代码刷新，解除90分钟断开限制.

// 使用方法：colab页面按下 F12或者 Ctrl+Shift+I (mac按 Option+Command+I) 在console（控制台） 输入以下代码并回车.

// 复制以下代码粘贴在浏览器console！！不要关闭浏览器以免失效
function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);