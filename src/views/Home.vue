<template>
 <div>
    <span v-if="seen">
      00000000000000000
      abcd {{ seen }}

    </span>
  </div>
</template>

<script>
export default {
  data () {
    return {
      seen: true,
      message: 'qwer'
    }
  }
}
</script>
