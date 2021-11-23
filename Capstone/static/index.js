var b = document.getElementById("btn");
var l = document.getElementById("login");
var r = document.getElementById("register");
function signup(){
	b.style.left = "110px";	
	l.style.left= "-450px";
	r.style.left= "50px";
}
function signin(){
	b.style.left = "0px";	
	l.style.left= "50px";
	r.style.left= "450px";
}
function myFunction() {
    var x = document.getElementById("passwordl");
    if (x.type === "password") {
      x.type = "text";
    } else {
      x.type = "password";
    }
  }
document.getElementById("login").addEventListener("submit",(event)=>{
    event.preventDefault()
})

document.getElementById("register").addEventListener("submit",(event)=>{
    event.preventDefault()
})

firebase.auth().onAuthStateChanged((user)=>{
    if(user){
        location.replace("Homepage.html")
    }
})

function Login(){
    const email = document.getElementById("emaill").value
    const password = document.getElementById("passwordl").value
    localStorage.setItem('email1',email);
    firebase.auth().signInWithEmailAndPassword(email, password)
    .catch((error)=>{
        document.getElementById("error1").innerHTML = error.message
    })
}

function Register(){
    var name = document.getElementById('name').value
    const email = document.getElementById("emailr").value
    var firebaseRef = firebase.database().ref('users/'+name)
    firebaseRef.set({
		Name : name,
		email : email
	});
    localStorage.setItem('email1',email);
    const password = document.getElementById("passwordr").value
    const confirmpassword = document.getElementById("confirmpassword").value
    if (password != confirmpassword) {document.getElementById("error2").innerHTML = "Password and ConfirmPassword are not same"; return;}
    firebase.auth().createUserWithEmailAndPassword(email, password)
    .catch((error) => {
        document.getElementById("error2").innerHTML = error.message
    });
}

function forgotPass(){
    const email = document.getElementById("emaill").value
    firebase.auth().sendPasswordResetEmail(email)
    .then(() => {
        alert("Reset link sent to your email id")
    })
    .catch((error) => {
        document.getElementById("error1").innerHTML = error.message
    });
}